load T;
load allbranch;
VT=[];VT1=[];
for t=1:128
A=allbranch;
A=[A;[[1:33]',34*ones(33,1)]];
if T(2,t)==0
    A(36,:)=[];
end
if T(1,t)==0
    A(31,:)=[];
end
if T(4,t)==0
    A(23,:)=[];
end
if T(3,t)==0
    A(21,:)=[];
end
if T(5,t)==0
    A(14,:)=[];
end
if T(7,t)==0
    A(12,:)=[];
end
if T(6,t)==0
    A(5,:)=[];
end
Cycles=grCycleBasis(A);

if size(Cycles,2)>0
    for i=1:size(Cycles,2)
        if sum(Cycles(size(A,1)-33+1:size(A,1),i))==0 
            VT1=[VT1,T(:,t)];
        end
    end
end

% if size(Cycles,2)>0
%     VT1=[VT1,T(:,t)];
% end
end

VT=[];
for i=1:size(T,2)
    if ismember(T(:,i)',VT1','rows')==0
        VT=[VT,T(:,i)];
    end
end
