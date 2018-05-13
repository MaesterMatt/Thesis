close all;
k = 10000; % k must be < min of all data

base1 = csvread('base.csv', 1, 0);
base2 = csvread('base2.csv', 1, 0);
base3 = csvread('base3.csv', 1, 0);

complete1 = csvread('complete.csv', 1, 0);
complete2 = csvread('complete2.csv', 1,0);
complete3 = csvread('complete3.csv', 1, 0);

base1 = base1(30:end-30, :);
base2 = base2(30:end-30, :);
base3 = base3(30:end-30, :);
complete1 = complete1(300:end-600, :);
complete2 = complete2(300:end-600, :);
complete3 = complete3(300:end-600, :);

base_v1 = linspace(0, 100, length(base1));
base_v2 = linspace(0, 100, length(base2));
base_v3 = linspace(0, 100, length(base3));

complete_v1 = linspace(0, 100, length(complete1));
complete_v2 = linspace(0, 100, length(complete2));
complete_v3 = linspace(0, 100, length(complete3));

base_v1 = [base_v1', base1(:,2)];
base_v2 = [base_v2', base2(:,2)];
base_v3 = [base_v3', base3(:,2)];

complete_v1 = [complete_v1', complete1(:,2)];
complete_v2 = [complete_v2', complete2(:,2)];
complete_v3 = [complete_v3', complete3(:,2)];

bv1 = base_v1(1:length(base_v1)/k:length(base_v1), :);
bv2 = base_v2(1:length(base_v2)/k:length(base_v2), :);
bv3 = base_v3(1:length(base_v3)/k:length(base_v3), :);

cv1 = complete_v1(1:length(complete_v1)/k:length(complete_v1), :);
cv2 = complete_v2(1:length(complete_v2)/k:length(complete_v2), :);
cv3 = complete_v3(1:length(complete_v3)/k:length(complete_v3), :);
   
b = (bv1(:,2) + bv2(:,2) + bv3(:,2))/3;
c = (cv1(:,2) + cv2(:,2) + cv3(:,2))/3;

plot(linspace(0, 100, k),b*0.393701); hold on;
plot(linspace(0, 100, k),c*0.393701);

legend('baseline', 'complete');
xlabel('Distance of Hallway in Feet');
ylabel('Distance from Right wall in Inches');