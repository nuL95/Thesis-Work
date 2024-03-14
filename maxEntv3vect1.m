addpath('C:\Users\Mark\Desktop\thesis research\matlab code\NEWMAXENT\dupelim')
rng('default')

%number of measurements
numMeas = 200;
%A random G matrix
N = 10;
p = .2;
p_shunt =.2 ;
edge_mean = 1;
edge_var = 0;
shunt_mean = 1;
shunt_var = 0;
numSources = 4;
G = GraphGeneration(N,numSources,p,p_shunt,edge_mean,edge_var,shunt_mean,shunt_var);

Test = retAdj(G);
D = DuplicationM(N);
E = EliminationM(N);
figure(2)
plot(graph(retAdj(G)))
i = randn(N,numMeas);
% i(numSources+1:end,:) = 0;
i(numSources+1:end,:) = 1e-7*randn(N-numSources,numMeas);
v = G^-1*i;
%
beta =2;
ghat = zeros(N,N);
    cvx_solver sedumi
    cvx_begin
    cvx_precision best
            variable g(N,N) symmetric
            X = g*v-i;
            X = X(:);
            A = kron(v',eye(N))*D;
            gv = E*g(:);
            maximize(trace(g)-beta*norm(g,'fro'))
            subject to
                for ii = 1:N
                    for jj = ii+1:N
                        g(ii,jj) <= 0;
                    end
                    g(ii,ii) >= 0;
                    sum(g(:,ii)) >= 0;
                end
                norm(X,1) <= 1e-10;


    cvx_end
G
gRec1 = full(g)
% gwr = WRound(gRec1,.1);
[prec,rec,fscore] = precision_recall_F(G,round(gRec1))
% [prec,rec,fscore] = precision_recall_F(G,gwr)

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

function G = GraphGeneration(N,S,p,shunt_p,edge_mean,edge_var,shunt_mean,shunt_var)

CM = abs(sqrt(edge_var)*randn(N,N)+ edge_mean);
CM = tril(CM,-1);
CM = CM+CM';

for ii = 1:N
    shunts(ii) = binRV(shunt_p);
end
if (sum(shunts) == 0)
    shunts(randi([1 N])) = 1;
end
for ii = 1:1000
    G = ER(N,p);
    if(checkConn(G) == 1)
        break;
    end
end

for ii = 1:S
    G(ii,ii) = 0;
    if (shunts(ii) ~= 1)
        if(-sum(G(:,ii)) == 1)
            G = addSourcedRandomConnection(G,ii);
        end
    end
end
for ii = S+1:N
    G(ii,ii) = 0;
%     if (shunts(ii) ~= 1)
        if(-sum(G(:,ii)) == 2)
            G = addRandomConnection(G,ii);
        end
        if(-sum(G(:,ii)) == 1)
            G = addTwoRandomConnection(G,ii);
        end
%     end
end

%%the mask is done here and the shunts vector can be used to determine
%%shunts but this next part is just to generate an unweighted graph.
G = CM.*G;
for ii = 1:N
    G(ii,ii) = -sum(G(ii,:))+(sqrt(shunt_var)*abs(randn())+shunt_mean)*shunts(ii);
%     G(ii,ii) = shunts(ii);
end

end

function Gnew = addSourcedRandomConnection(G,node)
    Gnew = G;
    N = length(G);
    rconn = randi([1 N]);
    index = find(G(node,:) == -1);
    for ii = 1:1e3
        if(rconn ~= index & rconn ~= node)
            break;
        end
        rconn = randi([1 N]);
    end
    Gnew(rconn,node) = -1;
    Gnew(node,rconn) = -1;
end

function Gnew = addRandomConnection(G,node)
Gnew = G;
    N = length(G);
    rconn = randi([1 N]);
    ind = find(G(node,:) == -1);
    for ii = 1:1e3
        if(rconn ~= ind(1) & rconn ~= node)
            if(rconn ~= ind(2) & rconn ~= node)
                break;
            end
        end
        rconn = randi([1 N]);
    end
    Gnew(rconn,node) = -1;
    Gnew(node,rconn) = -1;
end

function Gnew = addTwoRandomConnection(G,node)
    G1 = addSourcedRandomConnection(G,node);
    Gnew = addRandomConnection(G1,node);
end

function bin = checkConn(G)
    bin = 1;
    L = eig(G);
    if(L(2) <= 1e-7)
        bin = 0;
    end
end

function bin = binRV(p)
    bin = 1;
    x = rand;
    if(x >= p)
        bin = 0;
    end
end
function G = ER(N,p)
    G = zeros(N,N);
    for ii = 1:N
        for jj = ii+1:N
            G(ii,jj) = -binRV(p);
            G(jj,ii) = G(ii,jj);
        end
        G(ii,ii) = -sum(G(:,ii));
    end
end
function [precision,recall,f] = precision_recall_F(L_0,L)
% evaluate the performance of graph learning algorithms

L_0tmp = L_0-diag(diag(L_0));
edges_groundtruth = squareform(L_0tmp)~=0;

Ltmp = L-diag(diag(L));
edges_learned = squareform(Ltmp)~=0;

num_of_edges = sum(edges_learned);

if num_of_edges > 0
    [precision,recall] = perfcurve(double(edges_groundtruth),double(edges_learned),1,'Tvals',1,'xCrit','prec','yCrit','reca');
    if precision == 0 && recall == 0
        f = 0;
    else
        f = 2*precision*recall/(precision+recall);
    end
else
    precision = 0;
    recall = 0;
    f = 0;
end
end
function Gret = remEl(G,i,j)
Gret = G;
    if i == j
        Gret(i,i) = Gret(i,i) - sum(G(:,i));
    else
        Gret(i,j) = 0;
        Gret(j,i) = 0;
        Gret(i,i) = G(i,i) + G(i,j);
        Gret(j,j) = G(j,j) + G(i,j);
    end
end
function Gret = addEl(G,i,j,w)
    Gret = G; 
    if i == j
        Gret(i,i) = G(i,i) + w;
    else
        Gret(i,j) = -w;
        Gret(j,i) = -w;
        Gret(i,i) = G(i,i) + w;
        Gret(j,j) = G(j,j) + w;
    end
end
function A = retAdj(G)
    A = G;
    for ii = 1:length(G)
        A(ii,ii) = sum(G(:,ii));
    end
end
function Gre = WRound(G,threshold)
    for ii = 1:length(G)
        for jj = ii+1:length(G)
            if(abs(G(ii,jj)) > threshold)
                Gre(ii,jj) = -1;
                Gre(jj,ii) = -1;
            else
                Gre(ii,jj) = 0;
                Gre(jj,ii) = 0;
            end
        end
        diag = sum(G(ii,:));
        if (diag > threshold)
            Gre(ii,ii) = -sum(Gre(:,ii)) + 1;
        end
    end
end