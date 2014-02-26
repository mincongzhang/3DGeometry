function r()
close all
%source points
P_points = source_points ();
disp('read source points finished');
x1 = P_points(:,1);
y1 = P_points(:,2);
z1 = P_points(:,3);

scatter3(x1,y1,z1,5,[0 0 1],'.'); view([60,60,60]);drawnow;
set(gca, 'XLim', [-0.1 0.1]);
set(gca, 'YLim', [0 0.2]);
set(gca, 'ZLim', [0 0.1]);
hold on;

%target points
Q_points = target_points ();
disp('read target points finished');
x2 = Q_points(:,1);
y2 = Q_points(:,2);
z2 = Q_points(:,3);
scatter3(x2,y2,z2,5,[0 1 0],'.');drawnow;
hold on;

disp('press "space" to knn search...')
pause


%source P: blue; target Q: green; match P: red
sample_ratio = 100;
changed_P_points = MeshAlignment(P_points,Q_points,sample_ratio);
% x3 = changed_P_points(:,1);
% y3 = changed_P_points(:,2);
% z3 = changed_P_points(:,3);
% scatter3(x3,y3,z3,1,[1 0 0],'*'); drawnow;

for i=1:6
    disp(['Alignment times ',num2str(i+1)]);
    changed_P_points = MeshAlignment(changed_P_points,Q_points,sample_ratio);
    x3 = changed_P_points(:,1);
    y3 = changed_P_points(:,2);
    z3 = changed_P_points(:,3);
    scatter3(x3,y3,z3,1,[1 0 0],'*'); drawnow;
end
%     x3 = changed_P_points(:,1);
%     y3 = changed_P_points(:,2);
%     z3 = changed_P_points(:,3);
%     scatter3(x3,y3,z3,1,[1 0 0],'*'); drawnow;

end

function changed_P_points = MeshAlignment(P_points,Q_points,sample_ratio)
%%
%downsampling

sampled_P_points = P_points(1:sample_ratio:end,:);
sampled_Q_points = Q_points(1:sample_ratio:end,:);
Prow = size(sampled_P_points,1);
disp(['downsampling result ',num2str(Prow),' points']);

%%KNN search
matched_points = zeros(size(sampled_P_points));
count = 0;
for i = 1:Prow
    [n,d] = knnsearch(sampled_Q_points,sampled_P_points(i,:),'k',1);
    matched_points(i,:) = sampled_Q_points(n,:);
    if((i-1) == round(count*Prow))
    disp(['handling ',num2str(round(i/Prow*100)),'%'])
    count = count+0.1;
    end
end

% x3 = matched_points(:,1);
% y3 = matched_points(:,2);
% z3 = matched_points(:,3);
% scatter3(x3,y3,z3,5,[1 0 0],'*'); hold on;


%%
%mean values for each coordinates
P_mean = mean(sampled_P_points,1);
Q_mean = mean(matched_points,1);

%difference between mean values and each coordinates
[row col] = size(sampled_P_points);
P_diff = zeros(row,col);
Q_diff = zeros(row,col);
%method from lecture notes
C = zeros(3,3);
%method from slides
A = zeros(3,3);
A1 = zeros(3,3);
A2 = zeros(3,3);
for i=1:row
    P_diff(i,:) = sampled_P_points(i,:)-P_mean;
    Q_diff(i,:) = matched_points(i,:)-Q_mean;
    %method from lecture notes
    %C = C + P_diff(i,:)'*Q_diff(i,:);
    %method from slides
    A1 = A1 +Q_diff(i,:)'*P_diff(i,:);
    A2 = A2 +P_diff(i,:)'*P_diff(i,:);
end

%A = A1*(inv(A2));
A = A1;

%SVD
%method from lecture notes
%[U,S,V] = svd(C);
%method from slides
[U,S,V] = svd(A);
R = U*V';
T = Q_mean' - R*P_mean'; %3x1

%rotate source points
Prow = size(P_points,1);
for i = 1:Prow
    changed_P_points(i,:) = ( R*P_points(i,:)' + T )';
end

end

