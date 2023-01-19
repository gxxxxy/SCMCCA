function [X,lambda,err,stats] = MaxbetRtr(A,m,idxij, params)
% MaxbetRtr: Solve the MAXBET problem which has the following form：
%  max f(X)=tr(X^TAX) over the Cartesian product of m Stiefel manifolds M={X\in R^{n*l}|X{i}^TX{i}=I_l, X{i}\in R^{n_i*l_i}} 
% where A=(A{i,j}) \in R^{n*n}, and n=n_1+...+n_m. The size of X{i} is
% idxij{i}=[n_i,l_i].
%  
% This computes the MAXBET problem using the
% Riemannian Trust-Region with truncated CG inner solver.
%
% A should be a Hermitian matrix. 

%
% MaxbetRtr(A,idxij,params) allows the user to specify parameters that are passed 
% to the RTR solver.
%   params.X0        - initial iterate 
%   params.Delta_bar - maximum trust-region radius
%   params.Delta0    - initial trust-region radius
%   params.epsilon   - Outer Convergence tolerance (absolute)
%   params.useprec   - if non-zero, rtresgev will generate a LEFT preconditioner 
%                      for the problem. If params.useprec=1, the
%                      preconditioner is inv(diag(A)), otherwise it is the
%                      block diagonal (with triangular blocks) based on A.
%

% About: RTR - Riemannian Trust-Region
% (C) 2004-2007, P.-A. Absil, C. G. Baker, K. A. Gallivan
global TIMEMAX
tstart=tic;
% set pointers for functions required by RTR

   n = 0;
   for i=1:m
       n=n+size(A{i,1},1);
   end
   if nargin < 3
      params = [];
   end
   fns.R    = @(X,eta)R(m,X,eta);           % retraction
   fns.g    = @(X,eta,zeta)g(m,X,eta,zeta); % Riemannian metric
   fns.proj = @(X,eta)proj(m,X,eta); % projection onto tangent plane from nearby
   fns.f     = @(X)f(m,A,X);      % objective function
   fns.fgrad = @(X)grad(m,A,X);   % gradient of f
   fns.fhess = @(X,eta)H(m,A,X,eta); % Hessian of f
   fns.fhess_new = @(X,eta,lam)H_new(m,A,X,eta,lam); % Hessian of f
   fns.randT = @(X)randT(X);
   % set parameters for RTR algorithm
   % use tcg solve for now
   params.solver = 'tCG';
   d = n-m;    %max(idxij);
   if ~isfield(params,'max_inner') 
      params.max_inner = d;
   else
      params.max_inner = min(params.max_inner,d);
   end
   if isfield(params,'userandT') && params.userandT,
      fns.randT = @(X)randT(m,X); % random vector on tangent plane
   end
   if isfield(params,'useprec') && params.useprec,
       P=cell(m,1);
      if params.useprec==1
         % perform diagonal perturbation, K=inv(diag(A))=inv(P);
         for i=1:m
            P{i} = diag(-A{i});
         end
         %display('The diagonal preconditioner diag(A) is used.');
         fns.prec  = @(X,eta)precondP(P,m,X,eta); % precond with inv(Pi_x P Pi_x)
      else
         % perform a general block preconditioner with upper triangluar
         % diagonal blocks based on the Cholesky decompostion of A>0.
         for i=1:m
             P{i} = chol(-A{i,i});
         end
         %display('The block diagonal (with triangular blocks) based on chol(A) is used.');
         fns.prec  = @(x,eta)precondP(P,m,x,eta); % precond with inv(Pi_x P Pi_x)
      end
   end
   if ~isfield(params,'Delta_bar')
      params.Delta_bar = inf;
   end
   if ~isfield(params,'Delta0'),
      params.Delta0    = sqrt(3);
   end
   if ~isfield(params,'x0')
      params.x0=cell(m,1);
      for i=1:m
          [params.x0{i},~]=qr(randn(idxij{i}),0);
      end
   end
   Zero=cell(m,1);
   for i=1:m
       Zero{i}=zeros(idxij{i});
   end
   params.x0 = fns.R(params.x0,Zero);
   params.A = A;
  [X,stats] = rtr_multi_stiefel(fns,params);
   time=toc(tstart);
   if time>TIMEMAX
       X=0;
       lambda=0;
       err=0;
       stats=0;
       return;
   end
   %err=A*x;
   AX=cell(m,1);
   for i=1:m
       AX{i}=0;
       for j=1:m
           AX{i} = AX{i}+A{i,j}*X{j};
       end
   end
   lambda=cell(m,1);err=0;nrmx=0;
   for i=1:m
       lambda{i} = X{i}'*AX{i};
       lambda{i} = (lambda{i}+lambda{i}')/2;
       err=err+norm(AX{i}-X{i}*lambda{i},1);
       nrmx=norm(X{i},1);
   end
%    err=norm(err,1)/norm(x,1);
   err=err/nrmx;
   
   
   
function reta = R(m,X,eta)
reta=cell(m,1);
for i=1:m
    [reta{i},R]=qr(X{i}+eta{i},0);
    reta{i}=reta{i}* diag(sign(sign(diag(R))+.5)); %要求R的对角部分严格正，因此做一次符号运算
    %tmp=(eye(size(X{i},2))+eta{i}'*eta{i})^(-.5);
    %reta{i}=(X{i}+eta{i})*tmp;
    %[u,d,v]=svd(X{i}+eta{i}, 'econ');
    %reta{i}=u*v';
end

function ez = g(m,X,eta,zeta)
   ez=0;
   for i=1:m
       ez=ez+sum(sum(eta{i}.*zeta{i}));
   end


function reta = proj(m,X,eta)
reta=cell(m,1);   
for i=1:m
       tmp=X{i}'*eta{i};
       reta{i}=eta{i}-X{i}*(tmp+tmp')/2;
end

function f = f(m,A,X)
   f = 0;
   AX=cell(m,1);
   for i=1:m
       AX{i}=0;
       for j=1:m
           AX{i}=AX{i}+A{i,j}*X{j};
       end
        f=f+sum(sum((X{i}.*AX{i})));
   end

function grad = grad(m,A,X)
eta=cell(m,1);
for i=1:m
    eta{i}=0;
    for j=1:m
        eta{i}=eta{i}+2*A{i,j}*X{j};
    end
end
grad = proj(m,X,eta);

function Heta = H_new(m,A,X,eta,lambda)

   Aeta=cell(m,1);
   etalam=cell(m,1);
   for i=1:m
       Aeta{i}=0;
       for j=1:m
           Aeta{i}=Aeta{i}+A{i,j}*eta{j};
       end
       etalam{i}=eta{i}*lambda{i};
   end
   tmp=cell(m,1);
   for i=1:m
       tmp{i}=2*(Aeta{i}-etalam{i});
   end
   Heta= proj(m,X,tmp);

function Heta = H(m,A,X,eta)
   %eta=proj(m,X,eta);
   %pn=proj(idxij,x,A*pn);
   Aeta=cell(m,1);
   AX=cell(m,1);
   lambda=cell(m,1);etalam=cell(m,1);
   for i=1:m
       Aeta{i}=0;AX{i}=0;
       for j=1:m
           Aeta{i}=Aeta{i}+A{i,j}*eta{j};
           AX{i}=AX{i}+A{i,j}*X{j};
       end
       lambda{i}=X{i}'*AX{i};
       lambda{i}=(lambda{i}+lambda{i}')/2;
       etalam{i}=eta{i}*lambda{i};
   end
   tmp=cell(m,1);
   for i=1:m
       tmp{i}=2*(Aeta{i}-etalam{i});
   end
   Heta= proj(m,X,tmp);

function eta = randT(x)
   m = length(x);
   tmp=cell(m,1);
   for i=1:m
       tmp{i}=randn(size(x{i}));
   end
   eta = proj(m,x,tmp);
   