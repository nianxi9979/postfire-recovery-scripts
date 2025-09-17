clc;
clear;
datetime('now')

%% 

% 读取每个特征的CSV文件
mei_tem = readtable('normalized_mei_tem.csv');
mei_water = readtable('normalized_mei_water.csv');
mei_dnbr = readtable('normalized_mei_dnbr.csv');
mei_precip = readtable('normalized_mei_prec.csv');
mei_slope = readtable('normalized_mei_slope.csv');
mei_aspect = readtable('normalized_mei_aspect.csv');
mei_soil_type = readtable('meisoil.csv');
mei_vegetation_type = readtable('zhibei2.xlsx');
mei_sif = readtable('normalized_mei_sif.csv'); % 假设SIF是目标变量


% 为每个特征表格的列名添加前缀，以确保它们在合并时是唯一的
renamePrefixes = {'mei_tem_', 'mei_water_', 'mei_dnbr_', 'mei_precip_', 'mei_slope_', 'mei_aspect_', 'mei_soil_type_', 'mei_vegetation_type_'};
features_tables = {mei_tem, mei_water, mei_dnbr, mei_precip, mei_slope, mei_aspect, mei_soil_type, mei_vegetation_type};
feature_names = {'Temperature', 'Water', 'Fire Intensity', 'Precipitation', 'Slope', 'Aspect', 'Soil Type', 'Vegetation Type'};


% 检查所有表格的行数是否一致
numRows = height(mei_tem);
for i = 1:length(features_tables)
    if height(features_tables{i}) ~= numRows
        error('不一致的行数在表格 %d', i);
    end
end

% 合并特征数据
features = [];
for i = 1:length(features_tables)
    table_with_prefix = features_tables{i};
    table_with_prefix.Properties.VariableNames = strcat(renamePrefixes{i}, table_with_prefix.Properties.VariableNames);
    features = [features; table2array(table_with_prefix)]; % 使用table2array确保数据是数组形式
end

% 合并目标变量
y = table2array(mei_sif); % 确保y是数组形式

% 检查y的行数是否与features的行数一致
if size(y, 1) ~= size(features, 1)
    error('目标变量的行数与特征矩阵的行数不一致');
end

% 训练随机森林模型
nTrees = 100; % 树的数量
RFModel = TreeBagger(nTrees, features, y, 'Method', 'regression', 'OOBPredictorImportance', 'on');

% 获取特征重要性
importance = RFModel.OOBPermutedPredictorDeltaError;

% 绘制特征重要性图
figure;
bar(importance);
set(gca, 'XTick', 1:length(importance));
set(gca, 'XTickLabel', feature_names); % 使用特征名称
xtickangle(45); % 将x轴标签倾斜45度
xlabel('Feature');
ylabel('Importance');
title('SHAP Feature Importance');