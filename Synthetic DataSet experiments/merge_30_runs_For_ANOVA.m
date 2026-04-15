%% ============================================================
% دمج نتائج 30 ران كاملة بدون متوسط — كل Run صف مستقل
% ============================================================
clc; clear; close all;

folder = pwd;   % نفس موقع الملف
resultFolder = fullfile(folder, 'ANOVA_MERGE');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

files = dir(fullfile(folder, 'Run*.csv'));
fileNames = {files.name};
fileNumbers = regexp(fileNames, '\d+', 'match');
fileNumbers = cellfun(@(x) str2double(x{1}), fileNumbers);
[~, sortedIdx] = sort(fileNumbers);
files = files(sortedIdx);

numFilesToRead = min(30, length(files));
allData = table();

for k = 1:numFilesToRead
    T = readtable(fullfile(folder, files(k).name));
    T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);
    T.Run = repmat(k, height(T), 1);
    allData = [allData; T];
end

allData.ColName = strcat(string(allData.Imbalance), '%_', string(allData.Label), '%');

labelStrategies = {'uniform', 'nonuniform'};
imbalanceList = [1, 10, 30, 50];
labelList = [5, 10, 20, 50, 100];

desiredOrder = {};
for i = 1:length(imbalanceList)
    for j = 1:length(labelList)
        desiredOrder{end+1} = sprintf('%g%%_%g%%', imbalanceList(i), labelList(j)); %#ok<SAGROW>
    end
end

%% ============================================================
% إنشاء ملفين يحويان كل الرنز
% ============================================================
for s = 1:length(labelStrategies)
    strategy = labelStrategies{s};
    subset = allData(strcmpi(allData.LabelingStrategy, strategy), :);
    
    tempData = subset(:, {'Run', 'Dataset', 'DriftSpeed', 'ColName', 'GMean'});
    tempData.Properties.VariableNames{'DriftSpeed'} = 'Drift_Speed';
    
    % تحويل من long إلى wide مع بقاء كل Run
    pivot = unstack(tempData, 'GMean', 'ColName', 'VariableNamingRule', 'preserve');
    existingCols = intersect(desiredOrder, pivot.Properties.VariableNames, 'stable');
    pivot = [pivot(:, {'Run', 'Dataset', 'Drift_Speed'}), pivot(:, existingCols)];
    
    pivot = sortrows(pivot, {'Run', 'Dataset', 'Drift_Speed'});
    
    outFile = fullfile(resultFolder, sprintf('GMean_AllRuns_%s.xlsx', strategy));
    writetable(pivot, outFile);
    fprintf('✅ Saved: %s\n', outFile);
end

disp('=== DONE ===');
