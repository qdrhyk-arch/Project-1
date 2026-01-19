#   1.Import libraries
print('Libraries loading...')
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd 
import numpy as np
import sklearn.model_selection as sk_model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
print('Libraries loaded.')


#   2.Load data
print('Data loading...')
data = pd.read_csv('../data/input/ASD_AMD_code_data.csv', index_col=0)
total_data = data.copy()
del data
##  2.1为分层抽样做准备：对Y值进行分类（按数量级进行分段））
total_data.loc[(total_data['fik']>=1e-5)&(total_data['fik']<1e-4),'class'] = 'D'
total_data.loc[(total_data['fik']>=1e-4)&(total_data['fik']<1e-3),'class'] = 'E'
total_data.loc[(total_data['fik']>=1e-3)&(total_data['fik']<1e-2),'class'] = 'F'
total_data.loc[(total_data['fik']>=1e-2)&(total_data['fik']<1e-1),'class'] = 'G'
total_data.loc[(total_data['fik']>=1e-1)&(total_data['fik']<=1.5),'class'] = 'H'
##  2.2将未参与分类的少量样本删除
total_data.dropna(how='any',inplace=True)
print('Data loaded.')


#   3.Split data
print('Data splitting...')
##  3.1划分训练集
x = total_data.iloc[:,2:-1]
y = total_data.iloc[:,   1]
##  3.2分层采样
X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(x,y, test_size=0.3, stratify=total_data['class'])
del total_data
print('Data splitted.')


#   4.Functions
print('Functions loading...')
##  4.1定义基于RF的集成模型训练函数，包括基学习器RF的模型保存和预测结果保存
def RF_model(X_train, y_train, X_test, y_test, i):
    x = RandomForestRegressor(max_depth=None)
    x.fit(X_train, y_train)
    model_filename =f'RF_model_{i}.pkl'
    model_file_folder = '../data/output/f/models'
    if not os.path.exists(model_file_folder):
        os.makedirs(model_file_folder)
    with open(os.path.join(model_file_folder, model_filename), 'wb') as f:
        pickle.dump(x, f)
    y_pred_list_RF = pd.DataFrame(data=x.predict(X_test), index=y_test.index.values.tolist(), columns=['Predictions'])
    new_folder_path = '../data/output/f/prediction_data'
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    prediction_file_path = os.path.join(new_folder_path, f'RF_predictions_{i}.csv')
    try:
        y_pred_list_RF.to_csv(prediction_file_path)
        print(f"Prediction data saved to {prediction_file_path}")
    except Exception as e:
        print(f"Error saving prediction data: {e}") 
##  4.2定义合并CSV文件的函数   
def merge_csv_files(folder_path, output_file):
    combined_data = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                data = pd.read_csv(file_path, encoding=None, index_col=0)
                combined_data = pd.concat([combined_data, data], axis=1)
            except Exception as e:
                print(f"Reading {filename} failed with error: {e}")
    combined_data.to_csv(output_file, )
    print(f"Merged data saved to {output_file}")
##  4.3定义评估指标SMAPE计算函数
def smape_calculate(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)
##  4.4定义PAR计算函数
def PAR_calculation(y_true, y_pred, threshold):
    # Avoid division by zero and use absolute relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        re = np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))
    return np.sum(re <= threshold) / len(y_true) * 100
def evaluation_metrics_1(y_true, y_pred, absolute_threshold, relative_threshold):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # convert to percent
    smape = smape_calculate(y_true, y_pred)
    # Absolute error check
    count_ae = np.sum(np.abs(y_true - y_pred) <= absolute_threshold)/len(y_true) * 100
    # Relative error check (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)
        count_re = np.sum(rel_error <= relative_threshold)/len(y_true) * 100
    print(f'R2: {r2:.2f}\n'
        f'RMSE: {rmse:.2f}\n'  
        f'MAE: {mae:.2f}\n'  
        f'MAPE: {mape:.2f}%\n'  
        f'SMAPE: {smape:.2f}%\n'  
        f'Number of data within {absolute_threshold}: {count_ae:.2f}%\n'  
        f'Number of data within {relative_threshold*100}% relative error: {count_re:.2f}%')
    return r2, rmse, mae, mape, smape, count_ae, count_re
def evaluation_metrics_2(y_true, y_pred, relative_threshold):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # percent
    smape = smape_calculate(y_true, y_pred)
    par = PAR_calculation(y_true, y_pred, relative_threshold)
    return r2, rmse, mae, mape, smape, par
def classify_data_by_elements(X_input, Z_medium, Z_heavy, Z_super_heavy):
    data_low_index = X_input[X_input['Atomic number']<=Z_medium].index
    data_medium_index = X_input[(X_input['Atomic number']>Z_medium)&(X_input['Atomic number']<=Z_heavy)].index
    data_heavy_index = X_input[(X_input['Atomic number']>Z_heavy)&(X_input['Atomic number']<=Z_super_heavy)].index
    return data_low_index, data_medium_index, data_heavy_index
def save_elemental_predictions(y_true, y_pred, data_index, label1, label2, folder_path):
    y_pred_after = y_pred.loc[data_index]
    y_true_after = y_true.loc[data_index]
    data_atom = pd.DataFrame({f'y_true_{label1}': y_true_after, f'y_pred_{label1}': y_pred_after}, index=y_true_after.index)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f'RF-EL-C-{label1}-atom-{label2}.csv')
    data_atom.to_csv(file_path)
    return y_pred_after, y_true_after
print('Functions loaded.')



#   5.Main
print('Main process starting...')
############ 模型训练  ############
##  5.1基于RF的集成模型训练
print('Start training...')
if __name__ == '__main__':
    try:
        Parallel(n_jobs=4)(
            delayed(RF_model)(X_train, y_train, X_test, y_test, i) for i in tqdm(range(100), desc="Training models")
        )
        print("All models trained successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
print('Training done.')
############ 预测结果合并  ############
##  5.2合并预测结果
print('Merging predictions...')
folder_save_path = '../data/output/f/prediction_data/'  
output_file = '../data/output/f/total_predictions.csv'  
merge_csv_files(folder_save_path, output_file)
print('Merging predictions done.')

############ 最终预测结果计算及评估指标计算  ############
##  5.3计算最终预测结果及评估指标
print('Finishing final predictions...')
folder_path = '../data/output/f/'  
RF_predictions = pd.read_csv(folder_path+'total_predictions.csv', index_col=0)
predictions_df = RF_predictions.copy()
# Align existing test sets (from train_test_split) with the predictions
# Note: we intentionally use the previously created X_test/y_test from splitting
y_test = y_test.loc[predictions_df.index]
X_test = X_test.loc[predictions_df.index]
### 5.3.1定义分元素区段的原子序数
Z_medium = 20
Z_heavy = 53
Z_super_heavy = 100

############## 未嵌入置信控制的结果处理  ############
##  5.4未嵌入置信控制的结果
out_data_before = pd.DataFrame()
for i in predictions_df.index.values.tolist():
    df = predictions_df.loc[i]
    out_data_before.loc[i, 'mean'] = df.mean()
final_prediction_results = pd.DataFrame({'True Values':y_test, 'Predicted Values':out_data_before['mean'], }, index=predictions_df.index.to_list())
final_prediction_results.to_csv(folder_path+'RF-predictions-Bagging-0.csv', )  
y_pred_before = final_prediction_results['Predicted Values']
y_true_before = final_prediction_results['True Values']
### 5.4.1计算评估指标
r2_before, rmse_before, mae_before, mape_before, smape_before, _, count_re_before = evaluation_metrics_1(y_true_before, y_pred_before, 10, 0.1)
### 5.4.2保存评估指标
data_parameters_before_1 = pd.DataFrame({'R2': r2_before, 'RMSE': rmse_before, 'MAE': mae_before, 'MAPE': mape_before, 'SMAPE': smape_before,
                                        f'Number of data within 10% relative error':count_re_before}, index=[0])
data_parameters_before_1.to_csv(folder_path+'RF-predictions-Bagging-0-parameters.csv', )
############ 分元素区段测试 ############
### 5.4.3区分低中高Z原子预测结果保存及比例计算
X_input_before = X_test.loc[y_pred_before.index.to_list()]
data_light_index_before, data_medium_index_before, data_heavy_index_before = classify_data_by_elements(X_input_before, Z_medium, Z_heavy, Z_super_heavy)
### 5.4.4分别计算数据量
len_low_atom_before = len(data_light_index_before)
len_medium_atom_before = len(data_medium_index_before)
len_heavy_atom_before = len(data_heavy_index_before)
total_data_num = len(data_light_index_before) + len(data_medium_index_before) + len(data_heavy_index_before)
print(f'Proportion of low atom: {len_low_atom_before/total_data_num*100:.4f}%\n'
      f'Proportion of medium atom: {len_medium_atom_before/total_data_num*100:.4f}%\n'
      f'Proportion of heavy atom: {len_heavy_atom_before/total_data_num*100:.4f}%\n')
### 5.4.5分别保存预测结果
y_pred_light_before, y_true_light_before = save_elemental_predictions(y_true_before, y_pred_before, data_light_index_before, 'light', 'before', folder_path)
y_pred_medium_before, y_true_medium_before = save_elemental_predictions(y_true_before, y_pred_before, data_medium_index_before, 'medium', 'before', folder_path)
y_pred_heavy_before, y_true_heavy_before = save_elemental_predictions(y_true_before, y_pred_before, data_heavy_index_before, 'heavy', 'before', folder_path)
### 5.4.6保存总体预测结果
data_before = pd.DataFrame({
    'y_true': y_true_before,
    'y_true_low': y_true_light_before,
    'y_pred_low': y_pred_light_before,
    'y_true_medium': y_true_medium_before,
    'y_pred_medium': y_pred_medium_before,
    'y_true_heavy': y_true_heavy_before,
    'y_pred_heavy': y_pred_heavy_before,
})
data_before.to_csv(folder_path+'RF-predictions-Bagging-1-before.csv')
### 5.4.7分别计算各区段评估指标
r2_light_before, rmse_light_before, mae_light_before, mape_light_before, smape_light_before, par_light_before = evaluation_metrics_2(y_true_light_before, y_pred_light_before, 0.1)
r2_medium_before, rmse_medium_before, mae_medium_before, mape_medium_before, smape_medium_before, par_medium_before = evaluation_metrics_2(y_true_medium_before, y_pred_medium_before, 0.1)
r2_heavy_before, rmse_heavy_before, mae_heavy_before, mape_heavy_before, smape_heavy_before, par_heavy_before = evaluation_metrics_2(y_true_heavy_before, y_pred_heavy_before, 0.1)
### 5.4.8保存各区段评估指标
data_parameters_before_2 = pd.DataFrame({'R2': [r2_light_before, r2_medium_before, r2_heavy_before],
                                  'RMSE': [rmse_light_before, rmse_medium_before, rmse_heavy_before],
                                  'MAE': [mae_light_before, mae_medium_before, mae_heavy_before],
                                  'MAPE': [mape_light_before, mape_medium_before, mape_heavy_before, ],
                                  'SMAPE': [smape_light_before, smape_medium_before, smape_heavy_before, ],
                                  'PAR':[par_light_before, par_medium_before, par_heavy_before]},
                                  index=['light atom', 'medium atom', 'heavy atom',])
data_parameters_before_2.to_csv(folder_path+'RF-predictions-Bagging-0-parameters-0.csv', )
print('done.')
############### 嵌入置信控制的结果处理 ###############
##  5.5嵌入置信控制的结果处理
### 5.5.1采用四分位距的统计手段选取有效数据区间，排除极端值
out_data = pd.DataFrame()
for i in predictions_df.index.values.tolist():
    df = predictions_df.loc[i]
    q1 = np.quantile(df, 0.25)
    q3 = np.quantile(df, 0.75)
    iqr = q3 - q1
    down = float(q1-1.5*iqr)
    up = float(q3+1.5*iqr)
    df_2 = df[(df>=down)&(df<=up)] 
    out_data.loc[i, 'AE'] = np.abs(df_2.mean() - y_test[i])
    out_data.loc[i, 'RE'] = np.abs((df_2.mean() - y_test[i])/y_test[i])*100
    out_data.loc[i, 'IQR'] = iqr
    out_data.loc[i, 'RSD'] = (df_2.std())/(df_2.mean())
    out_data.loc[i, 'MEAN'] = df_2.mean()
print('Finishing correlation analysis...')
### 5.5.2三种相关系数矩阵的计算和保存
result1 = out_data[['AE','RE', 'RSD','IQR']].corr() # type: ignore
result2 = out_data[['AE','RE', 'RSD','IQR']].corr(method='spearman') # type: ignore
result1.to_csv(folder_path+'Pearson_correlation_result.csv')
result2.to_csv(folder_path+'Spearman_correlation_result.csv')
print('Correlation analysis done.')
### 5.5.3通过相对标准偏差RSD进行筛选
data_line_rsd = 0.01
data_retention_rate = out_data[(out_data['RSD']<data_line_rsd)].shape[0]/predictions_df.shape[0]
y_pred_rsd = out_data[(out_data['RSD']<data_line_rsd)]['MEAN']
y_true_rsd = y_test[y_pred_rsd.index.values.tolist()]
print('Final predictions done.')
# 保存集成结果
data_rsd = pd.DataFrame(data={'True Values': y_true_rsd,
                                 'Predicted Values': y_pred_rsd,
                                 'Absolute Error': y_true_rsd - y_pred_rsd,
                                 'Data retention rate': data_retention_rate})
data_rsd.to_csv(folder_path+'RF_data_wl_std.csv')
print('Final predictions saved.')
### 5.5.4计算评估指标
r2, rmse, mae, mape, smape, count_ae_after, count_re_after = evaluation_metrics_1(y_true_rsd, y_pred_rsd, 10, 0.1)
print(  
        f'r2:{r2:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAE: {mae:.4f}\n'
        f'MAPE: {mape:.4f}%\n'
        f'SMAPE: {smape:.4f}%\n'
        f'Number of data within 10% relative error: {count_re_after:.4f}%\n')
# 保存参数
data_parameters_after_1 = pd.DataFrame({'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SMAPE': smape,
                                  'Data retention rate':data_retention_rate, 'Number of data within 10nm':count_ae_after,
                                  f'Number of data within 10% relative error':count_re_after}, index=[0])
data_parameters_after_1.to_csv(folder_path+'RF-predictions-Bagging-1-parameters-0.csv', )
# 读取测试集对应的输入数据
X_input_after = X_test.loc[y_pred_rsd.index.to_list()]
# 分别获取数据
data_light_index_after, data_medium_index_after, data_heavy_index_after = classify_data_by_elements(X_input_after, Z_medium, Z_heavy, Z_super_heavy)
# 分别预测
y_pred_light_after, y_true_light_after = save_elemental_predictions(y_true_rsd, y_pred_rsd, data_light_index_after, 'light', 'after', folder_path)
y_pred_medium_after, y_true_medium_after = save_elemental_predictions(y_true_rsd, y_pred_rsd, data_medium_index_after, 'medium', 'after', folder_path)
y_pred_heavy_after, y_true_heavy_after = save_elemental_predictions(y_true_rsd, y_pred_rsd, data_heavy_index_after, 'heavy', 'after', folder_path)
data_after = pd.DataFrame({
    'y_true': y_true_rsd,
    'y_true_light': y_true_light_after,
    'y_pred_light': y_pred_light_after,
    'y_true_medium': y_true_medium_after,
    'y_pred_medium': y_pred_medium_after,
    'y_true_heavy': y_true_heavy_after,
    'y_pred_heavy': y_pred_heavy_after,
})
data_after.to_csv(folder_path+'RF-predictions-Bagging-1-after.csv')
r2_light_after, rmse_light_after, mae_light_after, mape_light_after, smape_light_after, par_light_after = evaluation_metrics_2(y_true_light_after, y_pred_light_after, 0.1)
r2_medium_after, rmse_medium_after, mae_medium_after, mape_medium_after, smape_medium_after, par_medium_after = evaluation_metrics_2(y_true_medium_after, y_pred_medium_after, 0.1)
r2_heavy_after, rmse_heavy_after, mae_heavy_after, mape_heavy_after, smape_heavy_after, par_heavy_after = evaluation_metrics_2(y_true_heavy_after, y_pred_heavy_after, 0.1)
# 计算数据量
print(f'data retention rate for low atom: {len(data_light_index_after)/len(X_input_after)*100:.4f}%')
print(f'data retention rate for medium atom: {len(data_medium_index_after)/len(X_input_after)*100:.4f}%')
print(f'data retention rate for heavy atom: {len(data_heavy_index_after)/len(X_input_after)*100:.4f}%')
# 计算数据量变化
len_low_atom_after = len(data_light_index_after)
len_medium_atom_after = len(data_medium_index_after)
len_heavy_atom_after = len(data_heavy_index_after)
print(f'data maintain change for low atom: {(len_low_atom_after-len_low_atom_before)/len_low_atom_before*100:.4f}%\n'
      f'data maintain change for medium atom: {(len_medium_atom_after-len_medium_atom_before)/len_medium_atom_before*100:.4f}%\n'
      f'data maintain change for heavy atom: {(len_heavy_atom_after-len_heavy_atom_before)/len_heavy_atom_before*100:.4f}%\n')
# 保存结果
data_parameters_after_2 = pd.DataFrame({'R2': [r2_light_after, r2_medium_after, r2_heavy_after, ],
                                  'RMSE': [rmse_light_after, rmse_medium_after, rmse_heavy_after, ],
                                  'MAE': [mae_light_after, mae_medium_after, mae_heavy_after, ],
                                  'MAPE': [mape_light_after, mape_medium_after, mape_heavy_after, ],
                                  'SMAPE': [smape_light_after, smape_medium_after, smape_heavy_after, ],
                                  'PAR':[par_light_after, par_medium_after, par_heavy_after]},
                                  index=['low atom', 'medium atom', 'heavy atom',])
data_parameters_after_2.to_csv(folder_path+'RF-predictions-Bagging-1-parameters-1.csv', )
print('Step.2 done.')
print('All done.')