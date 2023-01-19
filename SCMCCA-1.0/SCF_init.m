function [X] = SCF_init(A,X,index,opt)
%--------------------------------------------------------------------------
% SCF_init: Run a single SCF step twice with the starting point X
% to obtain the modified initial value X0.
%
% This initialization takes advantages of SCF and the algorithm follows the
% procedure in MA.[1]
%   
% Hongyi Du, January 19, 2023.
%
% [1]Xijun Ma, Chungen Shen, Li Wang, Lei-Hong Zhang, and Ren-Cang Li. A 
% self-consistent-field iteration for maxbet with an application to 
% multi-view feature extraction. Advances in Computational Mathematics, 
% 48(2):1–34, 2022.
%--------------------------------------------------------------------------

lobpcg_tol = opt.lobpcgtol; m=opt.m;

braA=A; % braA removes the diagonal block of A.
for i=1:m
    braA(index(i):index(i+1)-1,index(i):index(i+1)-1)=0;
end


% Run a single step SCF twice to modify initial value.  
for it = 1:2
    for i=1:m
        GX(index(i):index(i+1)-1,:) = braA(index(i):index(i+1)-1,:) * X;
        if index(i+1)-index(i) <= 500
            XGX=X(index(i):index(i+1)-1,:)* GX(index(i):index(i+1)-1,:)';
            EX = A(index(i):index(i+1)-1,index(i):index(i+1)-1) +XGX + XGX';
            [X_tmp,~] = mexeig_sort(EX);
            
            X_tmp = X_tmp(:,1:l);
        else
            try
                [X_tmp, ~, ~] = lobpcg(X(index(i):index(i+1)-1,:),...
                    @(Y)EkX(A(index(i):index(i+1)-1,index(i):index(i+1)-1),GX(index(i):index(i+1)-1,:),...
                    X(index(i):index(i+1)-1,:),Y),lobpcg_tol,50,0);
                if ~isreal(X_tmp)
                    XGX=X(index(i):index(i+1)-1,:)* GX(index(i):index(i+1)-1,:)';
                    AX1 = A(index(i):index(i+1)-1,index(i):index(i+1)-1) ...
                        + XGX+XGX';
                    AX1 = (AX1+AX1')/2; OPTS.isreal =1;
                    [X_tmp,~] = eigs(AX1,l,'LA',OPTS);
                end
            catch me
                XGX=X(index(i):index(i+1)-1,:)* GX(index(i):index(i+1)-1,:)';
                AX1 = A(index(i):index(i+1)-1,index(i):index(i+1)-1) ...
                    + XGX+XGX';
                AX1 = (AX1+AX1')/2; OPTS.isreal =1;
                [X_tmp,~] = eigs(AX1,l,'LA',OPTS);
            end
            
        end
        GtX=X_tmp'*GX(index(i):index(i+1)-1,:);
        [U, ~, V] = svd(GtX);
        X(index(i):index(i+1)-1,:)= X_tmp*(U*V');
    end
end

end

%--------------------------------------------------------------------------
function EX = EkX(Ak,Gk,Xk,Y)
EX = -Ak*Y - Gk *(Xk'*Y) - Xk *(Gk'*Y);
end