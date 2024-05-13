%%original problem
a=binvar(3,1);
F=[sum(a)<=2.5];
g=-sum(a);
optimize(F,g);

%%%p-cone  
a=sdpvar(3,1);
X=sdpvar(3,3);
% x=sdpvar(1,1);sum(a)<=2.5,eig([1,a';a,X])>=0
F=[0<=a<=1,0<=X<=1,norm([2.5*a-X*[1 1 1]'-(2.5-[1 1 1]*a)*[0.5 0.5 0.5]'],1000)<=(3)^(1/1000)/0.5*(2.5-[1 1 1]*a),diag(X)==a,X==X'];
g=-sum(a);
optimize(F,g);

%%%%%LS
a=sdpvar(3,1);
X=sdpvar(3,3);
% x=sdpvar(1,1);sum(a)<=2.5,eig([1,a';a,X])>=0
F=[0<=a<=1,diag(X)==a,X==X'];
for i=1:3
F=[F,2.5*a(i)-X(i,:)*[1 1 1]'>=0,(2.5-[1 1 1]*a)-(2.5*a(i)-X(i,:)*[1 1 1]')>=0];
end
g=-sum(a);
optimize(F,g);

%%BCC
a=sdpvar(3,1);
y=sdpvar(3,1);
% x=sdpvar(1,1);sum(a)<=2.5,eig([1,a';a,X])>=0
F=[0<=a<=1,0<=y<=1];
for i=1:3
F=[F,a(i)==y(i),2.5*a(i)-[1 1 1]*y>=0,(2.5-[1 1 1]*a)-(2.5*a(i)-[1 1 1]*y)>=0];
end
g=-sum(a);
optimize(F,g);