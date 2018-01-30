M1 = csvread('HallwayPush', 2, 1);
%M2 = csvread('Halldrive1', 2, 1);
[m1,n1] = size(M1);
%[m2,n2] = size(M2);
hallTestLength = 116; %feet
t1 = linspace(1,hallTestLength, m1);
%t2 = linspace(1,hallTestLength, m2);
xi = interp1(t1, M1(:,1), 1:hallTestLength);
%yi = interp1(t1, M1(:,2), 1:hallTestLength);
R = medfilt1(xi,1);
%L = medfilt1(yi,1);
plot(1:hallTestLength, R, '--'); hold on;
%plot(1:hallTestLength, L);
ylim([0, 200])