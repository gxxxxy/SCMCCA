function [X,info] = SCMCCA_aRTR(A,X,opt)
%--------------------------------------------------------------------------
% SCMCCA_aRTR: Solve the SCMCCA model which has the following form：
%  min f(X)=tr(X^TAX) over the Cartesian product of m Stiefel manifolds 
%  M={X\in R^{n*l}|X{i}^TX{i}=I_l, X{i}\in R^{n_i*l}}
% where A=(A{i,j}) \in R^{n*n}, and n=n_1+...+n_m. The size of X{i} is
% idxij{i}=[n_i,l]. In SCMCCA model, A = -(V^TV + \lambda P).
%
% This algorithm computes the SCMCCA model using the accelerated Riemannian
% Trust-Region solver, which applies RTR algorithm on product Stiefel 
% manifolds with SCF initialization and subspace acceleration.
%   
% Hongyi Du, January 19, 2023.
%--------------------------------------------------------------------------

[n,l] = size(X); % n,number of cells, l,dimension of integrated subspace

if nargin < 3
    opt = [];
end
% Initial setting
if ~isempty(opt)
    maxit = opt.maxit; tol_f = opt.tolf; tol_gn = opt.tolgn;
    idx=opt.idx; m=opt.m; tmax = opt.tmax;
else
    maxit =  60; tol_f = 1e-12; tol_gn = 1e-6;
    m = 5; idx =ones(m,1)*(n/m); tmax = 3;
end
index = zeros(m,1);
for i = 1 : m + 1
    index(i) = sum(idx(1:i-1))+1;
end

CellA=cell(m,m);
idCellA=cell(m,1);
for i=1:m
    for j=i:m
        CellA{i,j}=A(index(i):index(i+1)-1,index(j):index(j+1)-1);
        if j>i
            CellA{j,i}=CellA{i,j}';
        else
            CellA{j,j}=(CellA{j,j}+CellA{j,j}')/2;
        end
    end
    idCellA{i}=[size(CellA{i,1},1),l];
end

% Info about SCMCCA-aRTR (include initial point)
F = [];                 % Cost function of last two loops
info.f = [];            % Cost function of current loop
info.outiter = 1;       % Iteration of outer loop
info.diff_f = 1;        % Reduction of cost function
info.gradnorm = [];     % Gradient Norm of each loop (include initial point)
info.iter = 0;          % Iteration of each loop (include initial point)

Lam=cell(m,1);               % Lam_i = X_i^TA_iX.
AX = A*X; XAX = X'*AX;
info.cost = trace(XAX); % Cost function of each loop (include initial point)

for i=1:m
    Lam{i}=X(index(i):index(i+1)-1,:)'*AX(index(i):index(i+1)-1,:);
    Lam{i}=(Lam{i}'+Lam{i})/2;
    XLam(index(i):index(i+1)-1,:)=X(index(i):index(i+1)-1,:)*Lam{i};
end
info.gradnorm = norm(2*(AX-XLam),'fro');

% Modify the initialization using a single step SCF twice
X = SCF_init(A,X,index,opt);

% Update the start point information and initialize subspace S (Sub)
AX = A*X; XAX = X'*AX;
info.f = trace(XAX);
F = info.f;
Sub = X;

% Parameters to perform RTR each loop.
paramRTR.max_inner=20; paramRTR.max_outer=1;

% Parameters to perform RTR in the subspace
paramsub.max_inner=20; paramsub.max_outer=100;

while (info.outiter <= maxit) && (info.diff_f > tol_f) && (info.gradnorm(end) >= tol_gn)
% Stop criterion (satisfy either criterion to stop)
% (1) The maximum iteration step (maxit) is reached
% (2) The relative reduction of cost function (info.diff_f) is less than tol_f
% (3) The gradient norm is smaller than tol_gn
    
    % Calculate a subspace refinement in each loop 
    Q = cell(m,1); idQ = cell(m,1);
    for i=1:m
        [Q{i},~]=qr(Sub(index(i):index(i+1)-1,:),0);
        idQ{i}=[size(Q{i},2),l];
    end
    
    paramsub.x0 = cell(m,1); SubA = cell(m,m);
    for i=1:m
        paramsub.x0{i}=eye(idQ{i});
        for j=i:m
            SubA{i,j}=Q{i}'*A(index(i):index(i+1)-1,index(j):index(j+1)-1)*Q{j};
            if j>i
                SubA{j,i}=SubA{i,j}';
            else
                SubA{j,j}=(SubA{j,j}+SubA{j,j}')/2;
            end
        end
    end

    paramsub.epsilon=max(tol_gn,min(1e-5,info.gradnorm(end)/5));
    [Y,~,~,~] = MaxbetRtr(SubA,m,idQ, paramsub);
    for i=1:m
        X(index(i):index(i+1)-1,:)=Q{i}*Y{i};
    end
    
    % Parameter Update for RTR
    paramRTR.x0=cell(m,1);
    for i=1:m
        paramRTR.x0{i}= X(index(i):index(i+1)-1,:);
    end
    paramRTR.epsilon=max(tol_gn,min(1e-5,info.gradnorm(end)/5));

    [Y,~,~,~] = MaxbetRtr(CellA,m,idCellA, paramRTR);
    for i=1:m
        X(index(i):index(i+1)-1,:)=Y{i};
    end
    
    % Document numerical results in each loop
    AX = A*X;
    F = [F sum(sum(X.*AX))];
    info.f = F(end);
    info.diff_f = abs(info.f-F(end-1))/abs(info.f);
    info.iter = [info.iter;info.outiter];
    info.cost = [info.cost;info.f];

    % Calculate the gradient norm
    if mod(info.outiter,1) == 0
        for i=1:m
            Lam{i}=X(index(i):index(i+1)-1,:)'*AX(index(i):index(i+1)-1,:);
            Lam{i}=(Lam{i}'+Lam{i})/2;
            XLam(index(i):index(i+1)-1,:)=X(index(i):index(i+1)-1,:)*Lam{i};
        end
        tmp_gradnorm = norm(2*(AX-XLam),'fro');
        info.gradnorm = [info.gradnorm;tmp_gradnorm];

        if tmp_gradnorm <= tol_gn && (info.outiter > 1)
            return;
        end
    end

    if info.outiter<=tmax
        Sub=[Sub X];
    else
        Sub=[Sub X];
        Sub(:,1:l)=[];
    end
    info.outiter = info.outiter + 1;

    info.F = F;
end

end

