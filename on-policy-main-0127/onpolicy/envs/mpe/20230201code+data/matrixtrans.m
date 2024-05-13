function R=matrixtrans(v,EC)
EC=int16(EC);
AA=[];
for i=1:33
    tempA=zeros(33,37);
    for j=1:33
        if EC(j,i)~=0
            tempA(j,EC(j,i))=1;
        end
    end
    AA=[AA,tempA];
end
% s=sdpvar(37,1)
n=33;
b=repmat({v},n,1);
S=blkdiag(b{:});
R=AA*S;
