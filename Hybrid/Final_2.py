#!/usr/bin/env python
# coding: utf-8

# In[70]:


# (MODIFIED CELL 2) - Dask Import & Setup

import dask.dataframe as dd
import dask.array as da
import dask_ml.cluster as dkm
import dask_ml.decomposition as dkd
import dask_ml.preprocessing as dpr
import dask_ml.model_selection as dms
from dask.distributed import Client, LocalCluster
from implicit.als import AlternatingLeastSquares

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

import xgboost as xgb # Dask-XGBoost는 dask.dataframe을 직접 받음
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder # LabelEncoder는 Dask-ML에 없음 (대체 필요)

# --- 0. 하이퍼파라미터 (Dask용) ---
# (SAMPLE_FRAC은 1.0이므로 삭제)
SOURCE_FILE = 'df_master_preprocessed.parquet'
K_BEERS = 6
K_USERS = 8
N_COMPONENTS = 10 # Latent Factor 개수
THRESHOLD = 0.5   # (이전에 조정한 값)

# 경고 메시지 무시
warnings.filterwarnings('ignore')

print("Dask 및 Dask-ML 라이브러리 임포트 완료.")


# # (MODIFIED CELL 2) - Dask Client Start

# In[52]:


# (MODIFIED CELL 2) - Dask Client Start

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, wait # (★★★★★ wait 임포트 추가 ★★★★★)
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# Dask 클러스터를 로컬 머신에 설정합니다.
# (n_workers, memory_limit 등은 머신 사양에 맞게 조정하세요)
try:
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='16GB')
    client = Client(cluster)
    print("Dask Client가 시작되었습니다.")
    print(f"Dask 대시보드 링크: {client.dashboard_link}")
    print("위 링크를 브라우저에서 열어 Dask 작업 상태를 모니터링하세요.")
except Exception as e:
    print(f"Dask 클라이언트 시작 중 오류: {e}")
    print("Dask가 올바르게 설치되었는지 확인하세요.")

# --- 하이퍼파라미터 ---
SOURCE_FILE = 'df_master_preprocessed.parquet'


# # (MODIFIED CELL 3) - 840만 행 Dask로 로드 (샘플링 X)

# In[64]:


# (MODIFIED CELL 3) - 840만 행 Dask로 로드 (샘플링 X)

print(f"--- 1. 전체 데이터 로드 ({SOURCE_FILE}) ---")
# (★중요★) 840만 행 전체를 로드합니다. (아직 메모리에 올라가지 않음)
# engine='pyarrow'가 parquet 처리에 효율적입니다.
# 필요한 컬럼만 지정하면 속도가 더 빨라집니다.
cols_to_load = ['date', 'style', 'country_brewery', 'abv', 'smell', 'taste', 
                'feel', 'score', 'username', 'beer_id']

try:
    df_full = dd.read_parquet(
        SOURCE_FILE, 
        engine='pyarrow',
        columns=cols_to_load 
    )
    
    # npartitions를 조정하여 청크 크기 변경 가능 (예: repartition(npartitions=50))
    # df_full = df_full.repartition(npartitions=50) # (필요시 주석 해제)

    # Dask가 데이터를 메모리에 올려두고 재사용하도록 지시 (계산 속도 향상)
    df_full = df_full.persist() 
    
    print(f"전체 데이터 로드 준비 완료. (Dask lazy loading)")
    print(f"전체 데이터 행 수 (예상): {len(df_full)}") # .compute() 전이라도 len()은 빠르게 계산됨

except Exception as e:
    print(f"Dask로 Parquet 로드 중 오류 발생: {e}")
    print("Dask 또는 pyarrow 설치, 파일 경로를 확인하세요.")


# # (MODIFIED CELL 4) - Dask 피처 엔지니어링

# In[65]:


# (MODIFIED CELL 4) - Dask 피처 엔지니어링

print("\n--- 'style_group' 및 'geo_group' 피처 생성 중 (Dask) ---")

def group_style(style):
    if 'IPA' in str(style): return 'IPA'
    if 'Stout' in str(style): return 'Stout'
    if 'Ale' in str(style): return 'Ale'
    return 'Other'
    
def group_country(country):
    if country == 'US': return 'US'
    if country in ['DE', 'GB', 'BE']: return 'Europe'
    return 'Other'

# (★수정★) .apply -> .map_partitions
# Dask는 map_partitions를 사용할 때 출력 결과의 타입(meta)을 알려줘야 합니다.
df_full['style_group'] = df_full['style'].map_partitions(
    lambda s: s.apply(group_style), 
    meta=pd.Series(dtype='object', name='style_group')
)
df_full['geo_group'] = df_full['country_brewery'].map_partitions(
    lambda s: s.apply(group_country), 
    meta=pd.Series(dtype='object', name='geo_group')
)

print("'style_group', 'geo_group' 생성 완료 (Lazy).")


# # (MODIFIED CELL 5) - Dask Load Test (wait 함수 수정)

# In[66]:


# (MODIFIED CELL 5) - Dask Load Test (wait 함수 수정)

import time 

print(f"--- 1. Dask로 전체 데이터 로드 테스트 ({SOURCE_FILE}) ---")

if 'client' not in locals():
    print("!!! 오류: Dask Client가 실행되지 않았습니다. (이전 셀 실행 필요)")
else:
    try:
        # 1. 840만 행 전체를 Dask로 로드 (Lazy)
        cols_to_load = ['username', 'beer_id', 'score', 'date', 'style', 'country_brewery']
        
        df_full = dd.read_parquet(
            SOURCE_FILE, 
            engine='pyarrow',
            columns=cols_to_load
        )
        
        print(f"파일 읽기 준비 완료. (Lazy Loading)")
        
        # 2. (★핵심 테스트★) .persist()
        print("Dask .persist() 시작... (데이터를 메모리에 로드합니다. 시간이 걸릴 수 있습니다.)")
        start_time = time.time()
        
        df_full = df_full.persist() # (★계산 발생★)
        
        # 3. (★★★★★ 오류 수정 ★★★★★)
        # .persist()가 완료될 때까지 대기 (client.wait_for_futures -> wait)
        wait(df_full) 
        
        total_rows = len(df_full)
        
        print(f"--- 로드 테스트 성공! ---")
        print(f"로드 시간: {(time.time() - start_time):.2f} 초")
        print(f"총 {total_rows} 행이 Dask 메모리에 로드되었습니다.")
        
    except FileNotFoundError:
        print(f"!!! 오류: '{SOURCE_FILE}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"!!! Dask 로드(.persist()) 중 오류 발생: {e}")
        print("이제 'time' 또는 'wait' 오류가 아니라면, 이것이 진짜 메모리 부족/pyarrow 오류입니다.")


# # (MODIFIED CELL 4) - Dask 피처 엔지니어링

# In[67]:


# (MODIFIED CELL 4) - Dask 피처 엔지니어링

print("\n--- 'style_group' 및 'geo_group' 피처 생성 중 (Dask) ---")

# (df_full이 이전 Cell 5 테스트에서 로드되었다고 가정)
if 'df_full' not in locals() or not isinstance(df_full, dd.DataFrame):
    print("!!! 오류: df_full (Dask DataFrame)이 없습니다. 로드 셀을 다시 실행하세요.")
else:
    def group_style(style):
        # (style은 NA가 있어도 'IPA' in 'NA' (False)가 되어 안전함)
        if 'IPA' in str(style): return 'IPA'
        if 'Stout' in str(style): return 'Stout'
        if 'Ale' in str(style): return 'Ale'
        return 'Other'
        
    def group_country(country):
        # --- 💡 [수정된 부분] ---
        # "boolean value of NA" 오류를 피하기 위해 NA를 먼저 체크합니다.
        if pd.isna(country):
            return 'Other' # 또는 return pd.NA
        # ---
        if country == 'US': return 'US'
        if country in ['DE', 'GB', 'BE']: return 'Europe'
        return 'Other'

    # (★수정★) .apply -> .map_partitions
    df_full['style_group'] = df_full['style'].map_partitions(
        lambda s: s.apply(group_style), 
        meta=pd.Series(dtype='object', name='style_group')
    )
    df_full['geo_group'] = df_full['country_brewery'].map_partitions(
        lambda s: s.apply(group_country), 
        meta=pd.Series(dtype='object', name='geo_group')
    )
    
    # (★핵심★) .persist()로 CB 피처 계산 결과를 메모리에 유지
    df_full = df_full.persist()
    wait(df_full) # 계산 완료까지 대기
    
    print("'style_group', 'geo_group' 생성 및 Persist 완료.")


# # (MODIFIED CELL 5) - Latent Feature (SVD) (Dask)

# In[60]:


# --- 3-A. Latent Feature 추출 (SVD) (Dask) ---
print("--- 3-A. Latent Feature 추출 (SVD) (Dask) ---")

try:
    # --- 1. Dask DataFrame 준비 ---
    if 'df_full' not in locals():
         raise NameError("df_full (Dask DataFrame)이 없습니다. 이전 로드 셀을 실행하세요.")
    
    # df_full을 ddf 변수명으로 받아서 SVD 전용으로 사용
    ddf = df_full 
    
    # --- 2. NA/결측치 제거 ---
    # 피벗에 사용할 핵심 컬럼 3개('username', 'beer_id', 'score')의 NA를 제거
    print("피벗 대상 컬럼('username', 'beer_id', 'score')의 결측치(NA)를 제거합니다...")
    ddf = ddf.dropna(subset=['username', 'beer_id', 'score'])
    print("결측치 제거 완료 (Lazy).")

    # --- 3. categorize() 실행 ---
    print("Warning: 840만 행에 대한 categorize() 및 pivot_table은")
    print("         매우 느리고 Dask 클러스터 메모리를 초과할 수 있습니다.")

    print("Pivoting을 위해 'username'과 'beer_id'를 category 타입으로 변환합니다...")
    print("이 작업은 고유값(cardinality)이 많아 매우 오래 걸릴 수 있습니다.")
    
    ddf = ddf.categorize(columns=['username', 'beer_id'])
    
    # (선택 사항) 메모리가 충분하다면 persist() 사용
    # ddf = ddf.persist() 

    print("Category 타입 변환 완료 (Known). Pivot 시작...")

    # --- 4. pivot_table 실행 ('score' 사용) ---
    pivot_df = ddf.pivot_table(index='username', 
                               columns='beer_id', 
                               values='score') 

    # --- 5. SVD를 위한 .fillna(0) ---
    print("Pivot 테이블 생성 완료. SVD를 위해 NaN을 0으로 채웁니다...")
    pivot_df_filled = pivot_df.fillna(0)
    print("NaN 채우기 완료(Lazy). SVD 계산 시작...")
    
    # --- 💡 6. [수정된 부분] SVD 로직 실행 ---
    # (N_COMPONENTS는 Cell 37에서 10으로 정의됨)
    svd = dkd.TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    
    print("SVD .fit_transform() 계산 시작 (Dask Array 사용)...")
    
    # 
    # (💡💡💡 여기가 수정된 부분입니다 💡💡💡)
    # 'numblocks' 오류 해결: .values를 붙여 Dask Array로 변환
    #
    user_latent_features = svd.fit_transform(pivot_df_filled.values) # <--- .values 추가
    beer_latent_features = svd.components_
    
    # (SVD 결과를 다른 셀에서 사용하기 위해 DataFrame으로 변환)
    user_latent_df = pivot_df_filled.index.to_frame(name='username')\
                       .assign(**{f'user_latent_{i}': user_latent_features[:, i] for i in range(N_COMPONENTS)})
    beer_latent_df = pivot_df_filled.columns.to_frame(name='beer_id')\
                       .assign(**{f'beer_latent_{i}': beer_latent_features[i, :] for i in range(N_COMPONENTS)})

    # (결과를 메모리에 고정)
    user_latent_df = user_latent_df.persist()
    beer_latent_df = beer_latent_df.persist()
    wait(user_latent_df)
    wait(beer_latent_df)
    
    print("--- Dask SVD/Pivot 모든 작업 완료 ---")


except Exception as e:
    print(f"!!! Dask SVD/Pivot 중 오류 발생: {e}")
    print("--- 'category dtype' 오류가 아니라면, 이것이 진짜 메모리/성능 한계입니다. ---")


# # (MODIFIED CELL 4) - Dask 피처 엔지니어링???????

# In[36]:


# (MODIFIED CELL 4) - Dask 피처 엔지니어링

print("\n--- 'style_group' 및 'geo_group' 피처 생성 중 (Dask) ---")

# (df_full이 이전 Cell 5 테스트에서 로드되었다고 가정)
if 'df_full' not in locals() or not isinstance(df_full, dd.DataFrame):
    print("!!! 오류: df_full (Dask DataFrame)이 없습니다. 로드 셀을 다시 실행하세요.")
else:
    def group_style(style):
        if 'IPA' in str(style): return 'IPA'
        if 'Stout' in str(style): return 'Stout'
        if 'Ale' in str(style): return 'Ale'
        return 'Other'
        
    def group_country(country):
        if country == 'US': return 'US'
        if country in ['DE', 'GB', 'BE']: return 'Europe'
        return 'Other'

    # (★수정★) .apply -> .map_partitions
    df_full['style_group'] = df_full['style'].map_partitions(
        lambda s: s.apply(group_style), 
        meta=pd.Series(dtype='object', name='style_group')
    )
    df_full['geo_group'] = df_full['country_brewery'].map_partitions(
        lambda s: s.apply(group_country), 
        meta=pd.Series(dtype='object', name='geo_group')
    )
    
    # (★핵심★) .persist()로 CB 피처 계산 결과를 메모리에 유지
    df_full = df_full.persist()
    wait(df_full) # 계산 완료까지 대기
    
    print("'style_group', 'geo_group' 생성 및 Persist 완료.")


# In[29]:


# (MODIFIED CELL 6) - Beer Clustering (Dask-ML)

print("\n--- Beer Clustering (Dask-ML) ---")

# 1. 재료 준비
beer_features_df = df_full[['beer_id', 'style_group', 'geo_group']]\
    .drop_duplicates(subset=['beer_id'])\
    .dropna()\
    .set_index('beer_id') # set_index는 Dask에서 느린 작업

# 2. Dask-ML OneHotEncoding (pd.get_dummies 대체)
# (Dask-ML은 문자열 카테고리를 숫자로 먼저 변환해야 할 수 있음)
# (단순화를 위해 get_dummies를 쓰지만, 대용량에서는 Categorizer + OneHotEncoder 추천)
beer_features_processed = dd.get_dummies(
    beer_features_df.categorize(columns=['style_group', 'geo_group']),
    columns=['style_group', 'geo_group']
)

# 3. Dask-ML K-Means
kmeans_beer = dkm.KMeans(n_clusters=K_BEERS, random_state=42, n_init=10)
print("Beer K-Means 학습 시작 (Dask)...")

# (★계산 발생★) .fit_predict()는 계산을 트리거합니다 (메모리 주의)
beer_features_df['beer_cluster'] = kmeans_beer.fit_predict(beer_features_processed)
beer_features_df = beer_features_df.reset_index() # Merge를 위해 인덱스 리셋

print(f"Beer 클러스터링 완료. {K_BEERS}개 그룹으로 분류됨.")


# In[ ]:


# (MODIFIED CELL 7) - User Clustering (Dask-ML)

print("\n--- User Clustering (Dask-ML) ---")

# 1. 취향 벡터 (pd.crosstab 대체 -> pivot_table)
# crosstab(normalize='index')는 Dask로 매우 복잡함.
# (대안) Dask pivot_table로 합계(sum)를 구하고, map_partitions로 정규화(normalize)
style_affinity_pivot = df_full.pivot_table(
    index='username', columns='style_group', values='score', aggfunc='count'
).fillna(0)
# (정규화) 각 행(유저)의 합으로 나눔
user_style_affinity = style_affinity_pivot.map_partitions(
    lambda df: df.div(df.sum(axis=1), axis=0),
    meta=style_affinity_pivot._meta
).fillna(0)


# 2. 행동 기반 피처
user_numeric_features = df_full.groupby('username').agg(
    user_avg_score=('score', 'mean'),
    user_avg_abv=('abv', 'mean'),
    user_avg_smell=('smell', 'mean')
).fillna(0)

# 3. (★수정★) Dask Merge (pd.concat 대체)
# (SVD 셀이 성공했다는 가정 하에 user_latent_df 결합)
if 'user_latent_df' in locals():
    user_profile_df = dd.merge(user_numeric_features, user_style_affinity, on='username', how='outer')
    user_profile_df = dd.merge(user_profile_df, user_latent_df, on='username', how='left').fillna(0)
    print("User profile에 Latent Features 결합 (Lazy).")
else:
    user_profile_df = dd.merge(user_numeric_features, user_style_affinity, on='username', how='outer').fillna(0)

# 4. Dask-ML StandardScaler & KMeans
scaler_user = dpr.StandardScaler()
user_features_processed = scaler_user.fit_transform(user_profile_df.drop(columns=['username']))

kmeans_user = dkm.KMeans(n_clusters=K_USERS, random_state=42, n_init=10)
print("User K-Means 학습 시작 (Dask)...")

# (★계산 발생★) fit_predict 수행
user_profile_df['user_cluster'] = kmeans_user.fit_predict(user_features_processed)

print(f"User 클러스터링 완료. {K_USERS}개 그룹으로 분류됨.")


# In[ ]:


# (MODIFIED CELL 8) - Dask 최종 학습 데이터 생성

print("\n--- 최종 학습 데이터 생성 (Dask Merge) ---")

# 1. df_full + beer_cluster
df_model = dd.merge(df_full, beer_features_df[['beer_id', 'beer_cluster']], on='beer_id', how='left')

# 2. df_model + user_profile
user_cols_to_merge = ['username', 'user_cluster', 'user_avg_score']
user_latent_cols = [f'user_latent_{i}' for i in range(N_COMPONENTS)]
if 'user_latent_df' in locals():
    user_cols_to_merge.extend(user_latent_cols)

df_model = dd.merge(df_model, 
                    user_profile_df[user_cols_to_merge], 
                    on='username', 
                    how='left')
    
# 3. df_model + beer_latent
if 'beer_latent_df' in locals():
    df_model = dd.merge(df_model, beer_latent_df, on='beer_id', how='left')

# 4. Target 변수 생성
df_model['is_top_pick'] = (df_model['score'] > (df_model['user_avg_score'] + THRESHOLD)).astype(int)

print("'is_top_pick' 타겟 변수 생성 완료 (Lazy).")

# 5. (메모리 해제) Dask는 중간 변수들을 .persist()로 관리합니다.
# del df_full, user_profile_df, beer_features_df # (필요시)


# In[ ]:


# (MODIFIED CELL 9) - Dask Train/Test 분리

print("\n--- Train/Test/Validation 분리 (Dask-ML) ---")

features_to_use = [
    'smell', 'taste', 'feel',
    'style_group', 'geo_group',
    'beer_cluster', 'user_cluster'
]
if 'user_latent_df' in locals():
    features_to_use.extend([f'user_latent_{i}' for i in range(N_COMPONENTS)])
    features_to_use.extend([f'beer_latent_{i}' for i in range(N_COMPONENTS)])

target = 'is_top_pick'

# 1. Dask-ML LabelEncoder (Workaround)
# (Dask-ML에는 LabelEncoder가 없으므로, .categorize().get_dummies() 또는 .map_partitions 사용)
# 여기서는 편의상 .categorize()를 사용합니다.
categorical_features_final = ['style_group', 'geo_group', 'beer_cluster', 'user_cluster']
df_model = df_model.categorize(columns=categorical_features_final)
for col in categorical_features_final:
    df_model[col] = df_model[col].cat.codes # .cat.codes로 숫자형 변환

# 2. 결측치 제거
all_cols_needed = features_to_use + [target]
df_model = df_model.dropna(subset=all_cols_needed)
print(f"결측치 제거 (Lazy).")

X = df_model[features_to_use]
y = df_model[target]

# 3. Dask-ML Train/Test Split
X_train, X_test, y_train, y_test = dms.train_test_split(X, y, test_size=0.2, random_state=42)
X_train_sub, X_val, y_train_sub, y_val = dms.train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 4. (★중요★) 분리된 데이터를 메모리에 .persist()
# XGBoost 학습 시 데이터를 반복적으로 읽어야 하므로, 계산된 결과를 메모리에 유지시킵니다.
X_train_sub = X_train_sub.persist()
y_train_sub = y_train_sub.persist()
X_val = X_val.persist()
y_val = y_val.persist()
X_test = X_test.persist()
y_test = y_test.persist()

print(f"Train/Validation/Test 분리 및 Persist 완료.")


# In[ ]:


# (MODIFIED CELL 10) - XGBoost 학습 (Dask-XGBoost)

print("\n--- XGBoost Hybrid 모델 학습 시작 (Dask) ---")

# 1. 불균형 데이터 비율 계산
# (★계산 발생★) .compute()로 실제 값을 가져와야 합니다.
print("클래스 불균형 비율 계산 중...")
y_train_sub_computed = y_train_sub.compute() # (시간 소요)
ratio = (y_train_sub_computed == 0).sum() / (y_train_sub_computed == 1).sum()
print(f"scale_pos_weight ratio: {ratio:.2f}")

# 2. Dask XGBoost 모델 선언
# Dask 클라이언트를 모델에 알려줘야 합니다.
dask_model_xgb = xgb.dask.DaskXGBClassifier(
    client=client,
    objective='binary:logistic', eval_metric='auc',
    scale_pos_weight=ratio,
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    random_state=42,
    tree_method='hist' # Dask는 'hist'만 지원
)
    
# 3. Dask 데이터로 학습
print("Dask XGBoost 학습 시작...")
dask_model_xgb.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],      
    early_stopping_rounds=50,      
    verbose=100
)

print("모델 학습 완료.")


# In[ ]:


# (MODIFIED CELL 11) - 모델 평가 (Dask)

print("\n--- 모델 평가 (Test Set) ---")

# 1. Dask로 예측 (Lazy)
preds_proba_dask = dask_model_xgb.predict_proba(X_test)
preds_binary_dask = dask_model_xgb.predict(X_test)

# 2. (★계산 발생★) .compute()로 실제 결과(Numpy/Pandas)를 가져옴
print("Test Set 예측 결과 계산 중...")
preds_proba = preds_proba_dask[:, 1].compute() # 1 클래스 확률
preds_binary = preds_binary_dask.compute()
y_test_computed = y_test.compute()
print("계산 완료.")

# 3. Sklearn 지표로 평가
auc_score = roc_auc_score(y_test_computed, preds_proba)

print(f"\n[Hybrid Model - Test Set 결과]")
print(f"AUC (Area Under Curve): {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_computed, preds_binary))


# In[ ]:


# (MODIFIED CELL 12) - 랭킹 지표 평가 (P@K, R@K)

print("\n--- 랭킹 평가 지표 (Test Set) ---")

# 1. Dask에서 'username' 컬럼 가져오기 (이미 Persist됨)
usernames_test = X_test['username'].compute()

# 2. Pandas DataFrame으로 최종 결과 결합
test_results_df = pd.DataFrame({
    'username': usernames_test,
    'is_top_pick_true': y_test_computed,
    'probability_top_pick': preds_proba
})

print("P@K 계산 준비 완료:")
print(test_results_df.head())

# 3. (기존 P@K, R@K 계산 함수 사용 - 동일)
def calculate_ranking_metrics(df_results, k=10):
    # (기존과 동일한 함수 내용)
    # ...
    user_groups = df_results.groupby('username')
    user_metrics = {'precision@k': [], 'recall@k': [], 'map@k': []}
    
    for username, group in user_groups:
        total_true_positives = group['is_top_pick_true'].sum()
        if total_true_positives == 0:
            continue
        top_k_list = group.sort_values('probability_top_pick', ascending=False).head(k)
        hits_df = top_k_list[top_k_list['is_top_pick_true'] == 1]
        num_hits = len(hits_df)
        
        precision_at_k = num_hits / k
        recall_at_k = num_hits / total_true_positives
        user_metrics['precision@k'].append(precision_at_k)
        user_metrics['recall@k'].append(recall_at_k)
        
        if num_hits > 0:
            hit_ranks = (top_k_list.reset_index(drop=True).index + 1)[top_k_list['is_top_pick_true'] == 1]
            ap_sum = (pd.Series(range(1, num_hits + 1)) / hit_ranks).sum()
            average_precision = ap_sum / num_hits 
            user_metrics['map@k'].append(average_precision)
        else:
            user_metrics['map@k'].append(0.0)
            
    if not user_metrics['precision@k']:
        return pd.Series(index=['precision@k', 'recall@k', 'map@k'], data=[0.0, 0.0, 0.0])
        
    return pd.Series(user_metrics).apply(np.mean)

# 4. K=5, K=10일 때의 랭킹 지표 계산
k_5_metrics = calculate_ranking_metrics(test_results_df, k=5)
print(f"\n[Metrics @ K=5]\n{k_5_metrics}")
k_10_metrics = calculate_ranking_metrics(test_results_df, k=10)
print(f"\n[Metrics @ K=10]\n{k_10_metrics}")

