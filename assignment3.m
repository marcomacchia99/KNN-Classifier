%% clear workspace, remove figures and load data
clc
clear
close all

addpath('./mnist');

[training,labelTraining] = loadMNIST(0);
[test, labelTest] = loadMNIST(1);

%take two proportional subsets
subsetPercentage = 3;

subsetSizeTraining = round(size(training,1)*subsetPercentage/100);
randomIndexes = randperm(size(training,1),subsetSizeTraining);
trainingSubset = training(randomIndexes,:);
labelTrainingSubset = labelTraining(randomIndexes);

%take test subset proportional to training subset
subsetSizeTest = round(subsetSizeTraining*size(test,1)/size(training,1));
randomIndexes = randperm(size(test,1),subsetSizeTest);
testSubset = test(randomIndexes,:);
labelTestSubset = labelTest(randomIndexes);

%% try with different K value

%avoid k value that are divisible by the number of classes (10)
k=[1;2;3;4;5;9;15;19;29;39;49];
classification = zeros(size(testSubset,1),size(k,1));
errorRate = zeros(size(k,1),1);
for i=1:size(k,1)
    
    [classification(:,i),errorRate(i)]=kNNClassifier(trainingSubset,labelTrainingSubset,testSubset,k(i),labelTestSubset);
    
end

%% display error rate for each k value in bar graph

figure

bar(errorRate);
grid on;
title('Error rate related to k value');
ylabel('Error rate');
xlabel('Value of k');
%set xtick label to the correspondent k value
set(gca,'XTickLabel',k);
%analog to grid on, but shows only horizontal lines
set(gca, 'YGrid', 'on', 'XGrid', 'off');

%% on 10 tasks: each digit vs the remaining 9

nClasses = size(unique(labelTraining),1);
classes = sort(unique(labelTraining));

%stores the recognition rate (percentage of correct categorization) for
%each class
recognitionRate = zeros(nClasses,size(k,1));

%To compute the recognition rate, first extract from the classification
%matrix all the observation that corresponds effectively to the concerned
%class. Then find for each k how many time the observation has been
%correctly classified. Calculate the rate dividing this number for the
%number of observation that corresponds to the concerned class
for class=1:nClasses
    classIdx = find(labelTestSubset==classes(class));
    correctClass = sum(classification(classIdx,:)==classes(class));
    recognitionRate(class,:)=correctClass/size(classIdx,1);
end

%compute mean recognition rate
meanRecognitionRate = mean(recognitionRate,2);


%% display results

%display mean error rate
figure

bar(1-meanRecognitionRate);
grid on;
title('Mean error rate with different k when recognizing a class');
ylabel('Error rate');
xlabel('Class');
set(gca,'XTickLabel',classes);
set(gca, 'YGrid', 'on', 'XGrid', 'off');

%plot recognition rate using a 3d bar graph
figure
bar3(recognitionRate);
xlabel('Value of k');
ylabel('Class');
zlabel('Recognition rate');
set(gca,'XTickLabel',k);

%plot recognition rate for each k
figure
for i=1:size(k,1)
    subplot(3,4,i);
    bar(recognitionRate(:,i));
    grid on;
    title(['Recognition rate with k=',num2str(k(i))]);
    ylabel('Recognition rate');
    xlabel('Class');
    set(gca,'XTickLabel',classes);
    set(gca, 'YGrid', 'on', 'XGrid', 'off');
end
sgtitle('Recognition rate (with different values of k)');

%plot recognition rate for each class
figure
for i=1:nClasses
    subplot(3,4,i);
    bar(recognitionRate(i,:));
    grid on;
    title(['Recognition rate of class ',num2str(classes(i))]);
    ylabel('Recognition rate');
    xlabel('Value of k');
    set(gca,'XTickLabel',k);
    set(gca, 'YGrid', 'on', 'XGrid', 'off');
end
sgtitle('Recognition rate for each value of k (on the different classes)');
