#!/usr/bin/env python
# coding: utf-8

"""
Final_2_optimized.py
대용량 데이터셋(840만 행)을 위한 최적화된 하이브리드 추천 시스템

핵심 최적화:
1. Implicit ALS (SVD 대체) - 희소 행렬 직접 처리
2. 계층적 샘플링 - 활성 유저/인기 맥주 중심
3. Dask 병렬 처리 - Content-Based Features
4. 메모리 효율적 파이프라인
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings('ignore')

# Collaborative Filtering (희소 행렬 전용)
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix

# Content-Based & Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ==================== 0. 설정 ====================
SOURCE_FILE = 'df_master_preprocessed.parquet'
N_COMPONENTS = 10  # Latent Factors
K_BEERS = 6
K_USERS = 8
THRESHOLD = 0.5

# Dask 클러스터 시작
cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='16GB')
client = Client(cluster)
print(f"Dask 대시보드: {client.dashboard_link}")

# ==================== 1. 데이터 로드 (Dask) ====================
print("\n=== 1단계: 데이터 로드 ===")
df_full = dd.read_parquet(
    SOURCE_FILE,
    engine='pyarrow',
    columns=['date', 'style', 'country_brewery', 'abv', 'smell', 
             'taste', 'feel', 'score', 'username', 'beer_id']
)
print(f"전체 행 수: {len(df_full):,}")

# ==================== 2. 계층적 샘플링 (메모리 최적화) ====================
print("\n=== 2단계: 스마트 샘플링 ===")

# 활성 유저만 선택 (10개 이상 리뷰)
user_review_counts = df_full.groupby('username').size().compute()
active_users = user_review_counts[user_review_counts >= 10].index
print(f"활성 유저: {len(active_users):,}명")

# 인기 맥주만 선택 (50개 이상 리뷰)
beer_review_counts = df_full.groupby('beer_id').size().compute()
popular_beers = beer_review_counts[beer_review_counts >= 50].index
print(f"인기 맥주: {len(popular_beers):,}개")

# 필터링
df_filtered = df_full[
    df_full['username'].isin(active_users) & 
    df_full['beer_id'].isin(popular_beers)
]

# Pandas로 변환 (이제 메모리에 올릴 수 있음)
print("필터링된 데이터를 메모리에 로드 중...")
df_sample = df_filtered.compute()
print(f"최종 샘플 크기: {len(df_sample):,} 행")

# ==================== 3. Content-Based Features 생성 ====================
print("\n=== 3단계: Content-Based 피처 생성 ===")

# 3-1. Style & Geo Grouping
def group_style(style):
    if 'IPA' in str(style): return 'IPA'
    if 'Stout' in str(style): return 'Stout'
    if 'Ale' in str(style): return 'Ale'
    return 'Other'

def group_country(country):
    if pd.isna(country): return 'Other'
    if country == 'US': return 'US'
    if country in ['DE', 'GB', 'BE']: return 'Europe'
    return 'Other'

df_sample['style_group'] = df_sample['style'].apply(group_style)
df_sample['geo_group'] = df_sample['country_brewery'].apply(group_country)

# 3-2. Beer Clustering
print("Beer 클러스터링...")
beer_features_df = df_sample[['beer_id', 'style_group', 'geo_group']]\
    .drop_duplicates(subset=['beer_id'])\
    .dropna()\
    .set_index('beer_id')

beer_features_encoded = pd.get_dummies(beer_features_df)
kmeans_beer = KMeans(n_clusters=K_BEERS, random_state=42, n_init=10)
beer_features_df['beer_cluster'] = kmeans_beer.fit_predict(beer_features_encoded)
print(f"Beer 클러스터 분포:\n{beer_features_df['beer_cluster'].value_counts()}")

# ==================== 4. Collaborative Filtering (Implicit ALS) ====================
print("\n=== 4단계: CF - Implicit ALS (SVD 대체) ===")

# 4-1. Sparse Matrix 생성
ratings_df = df_sample[['username', 'beer_id', 'score']].dropna()

# ID를 숫자로 매핑
user_encoder = {u: i for i, u in enumerate(ratings_df['username'].unique())}
beer_encoder = {b: i for i, b in enumerate(ratings_df['beer_id'].unique())}

user_ids = ratings_df['username'].map(user_encoder).values
beer_ids = ratings_df['beer_id'].map(beer_encoder).values
scores = ratings_df['score'].values

# COO → CSR (ALS 입력 형식)
rating_matrix = coo_matrix(
    (scores, (user_ids, beer_ids)),
    shape=(len(user_encoder), len(beer_encoder))
).tocsr()

print(f"Rating Matrix: {rating_matrix.shape} (희소도: {rating_matrix.nnz / (rating_matrix.shape[0] * rating_matrix.shape[1]) * 100:.2f}%)")

# 4-2. ALS 학습
print("ALS 학습 시작...")
als_model = AlternatingLeastSquares(
    factors=N_COMPONENTS,
    regularization=0.01,
    iterations=15,
    use_gpu=False,  # GPU 있으면 True
    calculate_training_loss=True
)

als_model.fit(rating_matrix)
print("ALS 학습 완료!")

# 4-3. Latent Factors 추출
user_id_reverse = {i: u for u, i in user_encoder.items()}
beer_id_reverse = {i: b for i, b in beer_encoder.items()}

user_latent_df = pd.DataFrame(
    als_model.user_factors,
    columns=[f'user_latent_{i}' for i in range(N_COMPONENTS)]
)
user_latent_df['username'] = [user_id_reverse[i] for i in range(len(user_encoder))]

beer_latent_df = pd.DataFrame(
    als_model.item_factors,
    columns=[f'beer_latent_{i}' for i in range(N_COMPONENTS)]
)
beer_latent_df['beer_id'] = [beer_id_reverse[i] for i in range(len(beer_encoder))]

print(f"User Latent: {user_latent_df.shape}, Beer Latent: {beer_latent_df.shape}")

# ==================== 5. User Clustering ====================
print("\n=== 5단계: User 클러스터링 ===")

# 5-1. User Profile (행동 기반 + CF)
user_numeric = df_sample.groupby('username').agg(
    user_avg_score=('score', 'mean'),
    user_avg_abv=('abv', 'mean'),
    user_avg_smell=('smell', 'mean')
).fillna(0)

user_style_affinity = pd.crosstab(
    df_sample['username'], 
    df_sample['style_group'], 
    normalize='index'
)

user_profile_df = pd.concat([user_numeric, user_style_affinity], axis=1).fillna(0)
user_profile_df = user_profile_df.merge(user_latent_df, on='username', how='left').fillna(0)

# 5-2. K-Means
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_profile_df.drop(columns=['username']))

kmeans_user = KMeans(n_clusters=K_USERS, random_state=42, n_init=10)
user_profile_df['user_cluster'] = kmeans_user.fit_predict(user_features_scaled)
print(f"User 클러스터 분포:\n{user_profile_df['user_cluster'].value_counts()}")

# ==================== 6. 최종 학습 데이터 생성 ====================
print("\n=== 6단계: 최종 데이터 병합 ===")

df_model = df_sample.copy()

# Beer Cluster 병합
df_model = df_model.merge(
    beer_features_df[['beer_cluster']].reset_index(), 
    on='beer_id', 
    how='left'
)

# User Profile 병합
user_cols = ['username', 'user_cluster', 'user_avg_score'] + \
            [f'user_latent_{i}' for i in range(N_COMPONENTS)]
df_model = df_model.merge(
    user_profile_df[user_cols], 
    on='username', 
    how='left'
)

# Beer Latent 병합
df_model = df_model.merge(beer_latent_df, on='beer_id', how='left')

# Target 변수
df_model['is_top_pick'] = (
    df_model['score'] > (df_model['user_avg_score'] + THRESHOLD)
).astype(int)

print(f"최종 데이터: {df_model.shape}")
print(f"타겟 분포:\n{df_model['is_top_pick'].value_counts(normalize=True)}")

# 결측치 제거
features_to_use = [
    'smell', 'taste', 'feel',
    'beer_cluster', 'user_cluster'
] + [f'user_latent_{i}' for i in range(N_COMPONENTS)] + \
    [f'beer_latent_{i}' for i in range(N_COMPONENTS)]

df_model = df_model.dropna(subset=features_to_use + ['is_top_pick'])
print(f"결측치 제거 후: {len(df_model):,} 행")

# ==================== 7. 학습 ====================
print("\n=== 7단계: XGBoost 학습 ===")

X = df_model[features_to_use]
y = df_model['is_top_pick']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 범주형 피처 인코딩
from sklearn.preprocessing import LabelEncoder
cat_features = ['beer_cluster', 'user_cluster']
encoders = {}

for col in cat_features:
    le = LabelEncoder()
    all_values = pd.concat([
        X_train_sub[col].astype(str),
        X_val[col].astype(str),
        X_test[col].astype(str)
    ]).unique()
    
    le.fit(all_values)
    X_train_sub[col] = le.transform(X_train_sub[col].astype(str))
    X_val[col] = le.transform(X_val[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

# XGBoost 학습
ratio = (y_train_sub == 0).sum() / (y_train_sub == 1).sum()
print(f"Class Imbalance Ratio: {ratio:.2f}")

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=ratio,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)

# ==================== 8. 평가 ====================
print("\n=== 8단계: 평가 ===")

preds_proba = model.predict_proba(X_test)[:, 1]
preds_binary = model.predict(X_test)

auc_score = roc_auc_score(y_test, preds_proba)
print(f"\n[Test Set 결과]")
print(f"AUC: {auc_score:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, preds_binary))

# ==================== 9. 랭킹 지표 ====================
def calculate_ranking_metrics(df_results, k=10):
    user_groups = df_results.groupby('username')
    user_metrics = {'precision@k': [], 'recall@k': [], 'map@k': []}
    
    for username, group in user_groups:
        total_true = group['is_top_pick_true'].sum()
        if total_true == 0:
            continue
            
        top_k = group.sort_values('probability', ascending=False).head(k)
        hits = len(top_k[top_k['is_top_pick_true'] == 1])
        
        user_metrics['precision@k'].append(hits / k)
        user_metrics['recall@k'].append(hits / total_true)
        
        if hits > 0:
            hit_ranks = (top_k.reset_index(drop=True).index + 1)[top_k['is_top_pick_true'] == 1]
            ap = (pd.Series(range(1, hits + 1)) / hit_ranks).sum() / hits
            user_metrics['map@k'].append(ap)
        else:
            user_metrics['map@k'].append(0.0)
    
    return pd.Series(user_metrics).apply(np.mean)

# 테스트 데이터에 username 추가
test_results_df = df_model.loc[y_test.index, ['username']].copy()
test_results_df['is_top_pick_true'] = y_test
test_results_df['probability'] = preds_proba

print("\n[Ranking Metrics @ K=5]")
print(calculate_ranking_metrics(test_results_df, k=5))

print("\n[Ranking Metrics @ K=10]")
print(calculate_ranking_metrics(test_results_df, k=10))

# ==================== 10. 정리 ====================
client.close()
cluster.close()
print("\n완료! Dask 클러스터 종료.")

