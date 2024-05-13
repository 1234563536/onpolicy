clc;
clear;
%% basic data
load VT;
load EC;

a=zeros(7,1);
L=[0 0 0 0 0 1 1]'; %%line state
T=24;
pbuy=[0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.8 0.8 0.8 1.3 1.3 1.3 1.3 0.8 0.8 0.8 0.8 1.3 1.3 1.3 0.8 0.8 0.4];
psell=0.35*ones(1,24);

M=1e2;         %big M
% load B;
%%%independent storage node 
Is=[28 18];
%%%candidate sharing storage users 
Csu=[5 9 11 12 14 22 24 27 29 31];   %所有集合内的共享节点11的储能
%%%candidate sharing storage node
Csn=[5 11 27];

load RX;
load Cnode;
load allbranch;

BD=case33bw;
PD=BD.bus(:,3)/10;
QD=BD.bus(:,4)/10; %Demand settings
% PD(1)=1e-3;QD(1)=1e-3;
% PDT=repmat(PD,1,T)+0.0001*rand(33,T);
% QDT=repmat(QD,1,T)+0.0001*rand(33,T);

load FL;
load SLsum;
load SLl;
load SLu;
load PV;
load SLini;

co=0.4;
FL=co*(FL)/1e4;
SLsum=co*SLsum/1e4;SLl=co*SLl/1e4;SLu=co*SLu/1e4;PV=co*PV/1e4; SLini=SLini/1e4;  %转标幺值
PV=3*PV;
PV([1:4 6 7 8 25 30 31],:)=0;
QDL=0.1*FL;

R=RX;    
X=RX;
R(R==1)=BD.branch(:,3);
X(X==1)=BD.branch(:,4);   %R,X settings

%%  AL SL a s setting 
AL=[25 29;18 33;8 21;12 22;9 15;5 6;12 13];

%%  PL QL IL setting I 电流平方
P_l=sdpvar(37,T);Q_l=sdpvar(37,T);I_l=sdpvar(37,T);   %P,Q,I on the lines
for t=1:T
PL{t}=matrixtrans(P_l(:,t),EC);
QL{t}=matrixtrans(Q_l(:,t),EC);
IL{t}=matrixtrans(I_l(:,t),EC);
end

%%   Lstate BBL=binvar(7,1);  %Line state
  %Line state
Lstate=[0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0
0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	L(6,1)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0
0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	L(3,1)	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	1	0	0	0	0	L(5,1)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	L(7,1)	0	0	0	0	0	0	0	0	L(4,1)	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	L(2,1)
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	L(1,1)	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];


%% PDG QDG PSOP QSOP PG QG U settings   U 电压平方
P_g=sdpvar(1,T);Q_g=sdpvar(1,T);
P_dg=sdpvar(2,T);Q_dg=sdpvar(2,T);
% P_sop=sdpvar(4,1);Q_sop=sdpvar(4,1);
U=sdpvar(33,T);
for t=1:T
PDG{t}=[0	0	0	0	0	0	0	0	0	P_dg(1,t)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	P_dg(2,t)	0	0];
QDG{t}=[0	0	0	0	0	0	0	0	0	Q_dg(1,t)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	Q_dg(2,t)	0	0];
PG{t}=[P_g(t)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
QG{t}=[Q_g(t)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
end

SL=sdpvar(33,T);
aa=sdpvar(1,T);  bb=sdpvar(1,T); cc=sdpvar(1,T);  %Load connection indicatora=binvar(7,1);  %%%%%attacker 变量
%%%  Objective function

%%%%from the upper level
E_sc=0.2;E_28=0.1;E_18=0.05;
P_sc=0.05;P_28=0.05;P_18=0.05;
Esc_0=0;E28_0=0;E18_0=0;

Esc=sdpvar(1,T);
E28=sdpvar(1,T);
E18=sdpvar(1,T);

dis=binvar(12,T);
char=binvar(12,T);
Pdis=sdpvar(12,T);
Pchar=sdpvar(12,T);

dis_c=binvar(1,T);
char_c=binvar(1,T);
Pdis_c=sdpvar(1,T);
Pchar_c=sdpvar(1,T);

for t=1:T
DIS{t}=[0	0	0	0	dis(1,t)	0	0	0	dis(2,t)	0	dis(3,t)	dis(4,t)	0	dis(5,t)	 0	   0	0	dis(6,t)	0	0	0	dis(7,t)	0	dis(8,t)	0	0	dis(9,t)	dis(10,t)	dis(11,t)	0	dis(12,t) 0 0];
CHAR{t}=[0	0	0	0	char(1,t)	0	0	0	char(2,t)	0	char(3,t)	char(4,t)	0	char(5,t)	 0     0	0	char(6,t)	0	0	0	char(7,t)	0	char(8,t)	0	0	char(9,t)	char(10,t)	char(11,t)	0	char(12,t) 0 0];
PDIS{t}=[0	0	0	0	Pdis(1,t)	0	0	0	Pdis(2,t)	0	Pdis(3,t)	Pdis(4,t)	0	Pdis(5,t)	 0	   0	0	Pdis(6,t)	0	0	0	Pdis(7,t)	0	Pdis(8,t)	0	0	Pdis(9,t)	Pdis(10,t)	Pdis(11,t)	0	Pdis(12,t) 0 0];
PCHAR{t}=[0	0	0	0	Pchar(1,t)	0	0	0	Pchar(2,t)	0	Pchar(3,t)	Pchar(4,t)	0	Pchar(5,t)	 0	   0	0	Pchar(6,t)	0	0	0	Pchar(7,t)	0	Pchar(8,t)	0	0	Pchar(9,t)	Pchar(10,t)	Pchar(11,t)	0	Pchar(12,t) 0 0];
end

for t=1:T
    if t==1
        Esc(t)=Esc_0+0.95*Pchar_c(t)-Pdis_c(t)/0.95;
        E28(t)=E28_0+0.95*PCHAR{t}(28)-PDIS{t}(28)/0.95;
        E18(t)=E18_0+0.95*PCHAR{t}(18)-PDIS{t}(18)/0.95;
    else
        Esc(t)=Esc(t-1)+0.95*Pchar_c(t)-Pdis_c(t)/0.95;
        E28(t)=E28(t-1)+0.95*PCHAR{t}(28)-PDIS{t}(28)/0.95;
        E18(t)=E18(t-1)+0.95*PCHAR{t}(18)-PDIS{t}(18)/0.95;
    end
end

F=[];
F=[1e-6>=aa>=0,1e-6>=bb>=0,1e-6>=cc>=0];          %%%0<=sum(a)<=7,sum(a)==suma    
for t=1:T
   F=[F,0.93^2<=U(Cnode,t)<=1.07^2];
    F=[F,U(1,t)==1];
    F=[F,0<=P_dg(t)<=0.003,-0.9*P_dg<=Q_dg(t)<=0.9*P_dg];
    F=[F,norm([P_dg(1,t),Q_dg(1,t)])<=0.003];
    F=[F,norm([P_dg(2,t),Q_dg(2,t)])<=0.003];   %DG related constraints;
    F=[F,P_g(t)<=1,P_g(t)>=-1,Q_g(t)<=1,Q_g(t)>=-1];   %G related constraints
    %%%%%%%%%%%%%% constraints for P Q I on the lines;
   F=[F,-M*Lstate<=PL{t}<=M*Lstate,-M*Lstate<=QL{t}<=M*Lstate,0<=IL{t}<=M*Lstate,0<=IL{t}<=1];   %,G related constraints
    
%     for i=1:33
%         if i~=11
%             F=[F,-aa(t)<=(sum(PL{t}(:,i)-R(:,i).*IL{t}(:,i))+PG{t}(i)+PDG{t}(i)+PDIS{t}(i)-PCHAR{t}(i)-FL(i,t)-SL(i,t)+PV(i,t)-sum(PL{t}(i,:)))<=aa(t),...  %  最后的sum改了正负号
%                 -bb(t)<=(sum(QL{t}(:,i)-X(:,i).*IL{t}(:,i))+QG{t}(i)+QDG{t}(i)-QDL(i,t)-sum(QL{t}(i,:)))<=bb(t)];
%         else
%             F=[F,-aa(t)<=(sum(PL{t}(:,i)-R(:,i).*IL{t}(:,i))+PG{t}(i)+PDG{t}(i)+Pdis_c(t)-Pchar_c(t)-FL(i,t)-SL(i,t)+PV(i,t)-sum(PL{t}(i,:)))<=aa(t),...  %  最后的sum改了正负号   +PDIS{t}(i)-PCHAR{t}(i)
%                 -bb(t)<=(sum(QL{t}(:,i)-X(:,i).*IL{t}(:,i))+QG{t}(i)+QDG{t}(i)-QDL(i,t)-sum(QL{t}(i,:)))<=bb(t)];
%         end
%     end
    for i=1:33
        if i~=11
            F=[F,(sum(PL{t}(:,i)-R(:,i).*IL{t}(:,i))+PG{t}(i)+PDG{t}(i)+PDIS{t}(i)-PCHAR{t}(i)-FL(i,t)-SL(i,t)+PV(i,t)-sum(PL{t}(i,:)))==0,...  %  最后的sum改了正负号
                (sum(QL{t}(:,i)-X(:,i).*IL{t}(:,i))+QG{t}(i)+QDG{t}(i)-QDL(i,t)-sum(QL{t}(i,:)))==0];
        else
            F=[F,(sum(PL{t}(:,i)-R(:,i).*IL{t}(:,i))+PG{t}(i)+PDG{t}(i)+Pdis_c(t)-Pchar_c(t)-FL(i,t)-SL(i,t)+PV(i,t)-sum(PL{t}(i,:)))==0,...  %  最后的sum改了正负号   +PDIS{t}(i)-PCHAR{t}(i)
                (sum(QL{t}(:,i)-X(:,i).*IL{t}(:,i))+QG{t}(i)+QDG{t}(i)-QDL(i,t)-sum(QL{t}(i,:)))==0];
        end
    end
    %%%%%%%%%constraints for storage
    F=[F,PDIS{t}([Csu])<=dis_c(t)*P_sc,PDIS{t}([Csu])>=0,PCHAR{t}([Csu])<=char_c(t)*P_sc,PCHAR{t}([Csu])>=0];
    F=[F,Pdis_c(t)<=dis_c(t)*P_sc,Pdis_c(t)>=0,Pchar_c(t)<=char_c(t)*P_sc,Pchar_c(t)>=0,Esc(t)<=E_sc,Esc(t)>=0];
    F=[F,PDIS{t}(28)<=DIS{t}(28)*P_28,PDIS{t}(28)>=0,PCHAR{t}(28)<=CHAR{t}(28)*P_28,PCHAR{t}(28)>=0,E28(t)<=E_28,E28(t)>=0];
    F=[F,PDIS{t}(18)<=DIS{t}(18)*P_18,PDIS{t}(18)>=0,PCHAR{t}(18)<=CHAR{t}(18)*P_18,PCHAR{t}(18)>=0,E18(t)<=E_18,E18(t)>=0];
    F=[F,Pdis_c(t)==sum(PDIS{t}([Csu])),Pchar_c(t)==sum(PCHAR{t}([Csu]))];
%   F=[F,DIS{t}+CHAR{t}<=1,dis_c(t)+char_c(t)<=1];
   
    if t==1
        F=[F,Pdis_c(t)<=Esc_0,Pchar_c(t)<=E_sc-Esc_0,PDIS{t}(28)<=E28_0,PCHAR{t}(28)<=E_28-E28_0,PDIS{t}(18)<=E18_0,PCHAR{t}(18)<=E_18-E18_0];
    else
        F=[F,Pdis_c(t)<=Esc(t-1),Pchar_c(t)<=E_sc-Esc(t-1),PDIS{t}(28)<=E28(t-1),PCHAR{t}(28)<=E_28-E28(t-1),PDIS{t}(18)<=E18(t-1),PCHAR{t}(18)<=E_18-E18(t-1)];
    end
        
%     for v=1:size(allbranch,1)
%         i=allbranch(v,1);j=allbranch(v,2);
%         F=[F,-cc(t)<=(U(i,t)-2*(R(i,j)*PL{t}(i,j)+X(i,j)*QL{t}(i,j))+(R(i,j)^2+X(i,j)^2)*IL{t}(i,j)-U(j,t))<=cc(t)];
%         F=[F,norm([2*PL{t}(i,j),2*QL{t}(i,j),(U(i)-abs(IL{t}(i,j)))])<=(U(i,t)+abs(IL{t}(i,j)))];
%     end
    for v=1:size(allbranch,1)
        i=allbranch(v,1);j=allbranch(v,2);
        F=[F,(U(i,t)-2*(R(i,j)*PL{t}(i,j)+X(i,j)*QL{t}(i,j))+(R(i,j)^2+X(i,j)^2)*IL{t}(i,j)-U(j,t))==0];
        F=[F,norm([2*PL{t}(i,j),2*QL{t}(i,j),(U(i)-abs(IL{t}(i,j)))])<=(U(i,t)+abs(IL{t}(i,j)))];
    end
end
% F=[F,SL>=SLl,SL<=SLu];  %%%
F=[F,sum(SL,2)==SLsum,SL>=SLl,SL<=3*SLu];
F=[F,dis+char<=1,dis_c+char_c<=1];

g=0;%+1e2*(aa+bb+cc)   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:T
    for i=1:33
        if i~=11
            g=g-max(PG{t}(i)+PDG{t}(i)+PDIS{t}(i)-PCHAR{t}(i)-FL(i,t)-SL(i,t)+PV(i,t),0)*psell(t)...
                -min(PG{t}(i)+PDG{t}(i)+PDIS{t}(i)-PCHAR{t}(i)-FL(i,t)-SL(i,t)+PV(i,t),0)*pbuy(t);  %  最后的sum改了正负号
        else
            g=g-max(PG{t}(i)+PDG{t}(i)+PDIS{t}(i)-PCHAR{t}(i)+Pdis_c(t)-Pchar_c(t)-FL(i,t)-SL(i,t)+PV(i,t),0)*psell(t)...
                -min(PG{t}(i)+PDG{t}(i)+PDIS{t}(i)-PCHAR{t}(i)+Pdis_c(t)-Pchar_c(t)-FL(i,t)-SL(i,t)+PV(i,t),0)*pbuy(t);   %  最后的sum改了正负号
        end
    end
end
g=g*1e4;

%% optimization 
%optimize(F,g,sdpsettings('solver','mosek'));,'gurobi.QCPDual',1
tic;
optimize(F,g,sdpsettings('solver','gurobi','gurobi.QCPDual',1));
toc;

Re=cell(3,1);
Re{1,1}=[];
Re{2,1}=[value(P_dg),value(Q_dg)];
Re{3,1}=[value(U)];

for t=1:T
res28dis(t)=value(PDIS{t}(28));
res28char(t)=value(PCHAR{t}(28));
end

for t=1:T
res18dis(t)=value(PDIS{t}(18));
res18char(t)=value(PCHAR{t}(18));






end

resdisc=value(Pdis_c);
rescharc=value(Pchar_c);

P_g=value(P_g);
Esc=value(Esc);
E28=value(E28);
E18=value(E18);
SL=value(SL);

NLini=(FL+SLini-PV)';
NL18=value(FL(18,:)+SL(18,:)-PV(18,:)+Pchar(6,:)-Pdis(6,:))';
NL28=value(FL(28,:)+SL(28,:)-PV(28,:)+Pchar(10,:)-Pdis(10,:))';
NLscini=sum(NLini(:,[Csu]),2);
NLsc=value(sum(FL([Csu],:))+sum(SL([Csu],:))-sum(PV([Csu],:))+sum(Pchar([1 2 3 4 5 7 8 9 11 12],:))-sum(Pdis([1 2 3 4 5 7 8 9 11 12],:)));
