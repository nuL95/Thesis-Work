clear
rng("default");
s = rng;
%The range of traces to sweep through
traceMax = 30:40;
%number of measurements
numMeas = 100;
%A random G matrix
N = 10;
G = [4	0	0	-1	0	-1	0	-1	0	-1
0	2	0	-1	-1	0	0	0	0	0
0	0	4	0	0	0	-1	-1	-1	-1
-1	-1	0	5	-1	-1	0	0	0	-1
0	-1	0	-1	4	-1	0	0	0	0
-1	0	0	-1	-1	4	0	0	0	0
0	0	-1	0	0	0	3	0	-1	-1
-1	0	-1	0	0	0	0	3	0	-1
0	0	-1	0	0	0	-1	0	3	-1
-1	0	-1	-1	0	0	-1	-1	-1	6]


onNodes = 1:7;

[v, i] = generateSignal(G,onNodes,numMeas);
%
for n = 1:length(traceMax)
    cvx_begin quiet
            variable g(N,N) symmetric
            X = g*v-i;
            X=X(:);
            minimize(norm(X,2))
            subject to
                for ii = 1:N

                    for jj = ii+1:N
                        g(ii,jj) <= 0;
                        g(ii,jj) >= -1;
                    end
                    sum(g(:,ii)) >= 0;
                end
                trace(g) == traceMax(n);
    cvx_end
    if cvx_optval <= .001
        ghat = g;
    end
    opval(n) = cvx_optval;
end

figure(1)
plot(traceMax,opval)
gRec = full(ghat)
G
%

reffMat(G)
reffMat(ghat)
function [G] = GenG(Size,LowerLimit,UpperLimit,sparse)
%Generates a random connectivity matrix of SizexSize dimensions of random
%uniformly distributed interger values between LowerLimit and UpperLimit
%   Generate a random lower triangle matrix, add it to the transpose to get a symmetric matrix, then
%   sum of the rows and randomly add (50% chance to add) another integer to
%   the diagonal for the shunt (self loop)
    Gup = -tril(randi([LowerLimit UpperLimit], Size,Size),-1);
    for i = 1:sparse
        Gup = Gup.*tril(randi([0 1], Size,Size),-1);
    end
    G = Gup + Gup';

    for ii = 1:Size
        G(ii,ii) = -sum(G(ii,:))+randi([1 1])*randi([LowerLimit UpperLimit]);
    end
end

function [V,I] = generateSignal(G,onNodes,numMeas)
 N = length(G);
 onvec = zeros(1,N);
 I = zeros(N,numMeas);
 V = zeros(N,numMeas);
 for n = 1:length(onNodes)
     onvec(onNodes(n)) = 1;

 end
 offvec = ones(1,N)-onvec;
 offNodes = [];
 for n = 1:N
    if offvec(n) == 1
        offNodes = [offNodes n];
    end
 end

 for n = 1:numMeas
     i = zeros(N,1);
    v = randn(1,N).*onvec;
    ip = G*v';
    A = G*diag(offvec);
    A(:,all(A == 0))=[];
    B = diag(onvec);
    B(:,all(B == 0))=[];
    C = [A -B -ip];
    D = rref(C);
    for k = 1:length(offNodes)
        v(offNodes(k)) = D(k,end);
    end
    for k = 1:length(onNodes)
        i(onNodes(k)) = D(k+length(offNodes),end);
    end
    V(:,n) = v';
    I(:,n) = i;
 end

end

function Gunpacked = unpackG(G)
N = length(G);
Gunpacked = zeros(N+1,N+1);
Gunpacked(1:end-1,1:end-1) = G;
for i = 1:N
    Gunpacked(i,end) = -sum(G(i,:));
    Gunpacked(end,i) = Gunpacked(i,end);
    Gunpacked(i,i) = 0;
end
end

function [Reff,Rtot] = reffMat(G)
    N = length(G);
    Reff = zeros(N,N);
    Ginv = G^-1;
    for r = 1:N
        for c = 1:N
            if(r == c)
                Reff(r,c) = Ginv(r,c);
            else
                Reff(r,c) = Ginv(r,r)+Ginv(c,c)-2*Ginv(r,c);
            end
        end
    end
    0.5*ones(N,1)'*Reff*ones(N,1);
end

function Gred = kr(G,startIndex,endIndex)
%matrix should have indices from start to end be the boundary nodes
    size = length(G);
    Gss = G(startIndex:endIndex,startIndex:endIndex);
    Gos = G(startIndex:endIndex,endIndex+1:size);
    Goo = G(endIndex+1:size,endIndex+1:size);
    Gred = Goo-Gos'*inv(Gss)*Gos;
end