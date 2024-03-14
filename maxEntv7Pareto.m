addpath('C:\Users\Mark\Desktop\thesis research\matlab code\NEWMAXENT\dupelim')
rng('default')
clear
%% Creation of Graph
numMeas = 200;
N = 10;
p = .6;
p_shunt =.6 ;
edge_mean = 1;
edge_var = 0;
shunt_mean = 1;
shunt_var = 0;
numSources = 4;
G = GraphGeneration(N,numSources,p,p_shunt,edge_mean,edge_var,shunt_mean,shunt_var);
D = DuplicationM(N);
E = EliminationM(N);
figure(2)
plot(graph(retAdj(G)))
%% Signal Generation
i = randn(N,numMeas);
i(numSources+1:end,:) = 1e-7*randn(N-numSources,numMeas);

v = G^-1*i;
%% Pareto Front Parameters
nvars = N*(N+1)/2;
ID = eye(N);
Aeq = kron(v',ID)*D;
A = kron(ID,ones(N,1)*ones(1,N))*D;
A = -A(1:N:end,:);
beq = i(:);
b = zeros(N,1);
c = diag(E*ID(:));
c( all(~c,2), : ) = [];
pL = sum(c);
diagInd = find(pL);
lb = zeros(N*(N+1)/2,1);
ub = zeros(N*(N+1)/2,1);
for ii = 1:nvars
    if(pL(ii) == 1)
        lb(ii) = 0;
        ub(ii) = inf;
    else
        lb(ii) = -inf;
        ub(ii) = 0;
    end
end
obj = @(g)[g(diagInd)];
% options.ParetoSetChangeTolerance = 1e-6;
options.ConstraintTolerance = 1e-8;
psize = 60;
options.ParetoSetSize = psize;
% options.MaxFunctionEvaluations = 100e3

%
% beta = 0;
%     cvx_solver sedumi
%     cvx_begin
%             variable g0(N,N) symmetric
%             X = g0*v-i;
%             X = X(:);
%             gv = E*g0(:);
%             maximize(trace(g0)-beta*norm(g0,'fro'))
%             subject to
%                 for ii = 1:N
%                     for jj = ii+1:N
%                         g0(ii,jj) <= 0;
%                     end
%                     sum(g0(:,ii)) >= 0;
%                 end
%                 norm(X,1) <= 1e-12;
% 
% 
%     cvx_end
% 
% [prec,rec,fscore] = precision_recall_F(G,round(g0))
% options.InitialPoints.X0 = (E*g0(:))';
% options.InitialPoints.Fvals = nonzeros(g0.*ID)';
[g,objval,exitflag,output,residuals] = paretosearch(obj,nvars,A,b,Aeq,beq,lb,ub,[],options)
% g = gamultiobj(obj,nvars,A,b,Aeq,beq,lb,ub)
% ovInd = 0;
% for ii = 1:psize
%     if(objval(ii,:) > 0.5)
%         ovInd = [ovInd ii];
%     end
% end
%To pick which g is recovered, 'end' is the index that selects the point on
%the pareto front.
gRec = reshape(D*g(end,:)', [N N])
G
[prec,rec,fscore] = precision_recall_F(G,round(gRec))

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
    G(ii,ii) = -sum(G(ii,:))+(sqrt(shunt_var)*randn()+shunt_mean)*shunts(ii);
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

function A = retAdj(G)
    A = G;
    for ii = 1:length(G)
        A(ii,ii) = sum(G(:,ii));
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