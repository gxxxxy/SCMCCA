function [P,lambda] = mexeig_sort(X)
%  Rearrange lambda and P in the nonincreasing order

[P,lambda] = mexeig(X);
P          = real(P);
lambda     = real(lambda);
if issorted(lambda)
    lambda = lambda(end:-1:1);
    P      = P(:,end:-1:1);
elseif issorted(lambda(end:-1:1))
    return;
else
    [lambda, Inx] = sort(lambda,'descend');
    P = P(:,Inx);
end

end

