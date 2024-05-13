classdef environment < handle
    properties
        N; % 点数
        B; % 边数
        gamma_wheel;   %  %单位：元/10MW
        ans;
        allbranch; % 包括branch所有边的起始节点及对应的电阻电抗(118m文件中branch的前4列)
    end
    
    methods
        % 初始化模块
        function df = environment()
            df.N = 118;
            df.B = df.N-1;
            df.gamma_wheel = 140;
            df.ans = case118zh;
            branchdata = df.ans.branch;
            df.allbranch = branchdata(:,1:4);
        end        
        %% 潮流计算模块
        function [QL,R,IL,V,X,Pg] = Calc_Distflow(T, PN, QN, df)
        %%%%% 输入
            % T: 所有边的状态，1为连接，0为断开, shape [1, -1]
            % PN: 节点有功负荷
            % QN: 节点无功负荷
        %%%%% 输出
            % QL: 文档中的Q
            % R: 支路电阻
            % IL: 支路电流，文档中的L
            % V: 节点电压
            % X: 支路电抗
            % Pg 平衡节点注入的功率（电源提供的功率） 
            yalmip('clear')
            allb = df.allbranch(find(T~=0),:);
            % PL QL IL setting I 电流平方
            % P,Q,I on the lines,顺序和branch里的一致
            P_l = sdpvar(df.B,1);
            Q_l = sdpvar(df.B,1);
            I_l = sdpvar(df.B,1);
            %
            R = zeros(df.B, df.N);
            X = zeros(df.B, df.N);
            
            for i=1:size(allb,1)
                if allb(i,1)<allb(i,2)
                    PL(allb(i,1),allb(i,2))=P_l(i);
                    QL(allb(i,1),allb(i,2))=Q_l(i);
                    IL(allb(i,1),allb(i,2))=I_l(i);
                    R(allb(i,1),allb(i,2))=allb(i,3);
                    X(allb(i,1),allb(i,2))=allb(i,4);
                else
                    PL(allb(i,2),allb(i,1))=P_l(i);
                    QL(allb(i,2),allb(i,1))=Q_l(i);
                    IL(allb(i,2),allb(i,1))=I_l(i);
                    R(allb(i,2),allb(i,1))=allb(i,3);
                    X(allb(i,2),allb(i,1))=allb(i,4);
                end
            end
            
            if size(PL,1) < df.N
                PL=[PL;zeros(df.N-size(PL,1),df.N)];
                QL=[QL;zeros(df.N-size(QL,1),df.N)];
                IL=[IL;zeros(df.N-size(IL,1),df.N)];
                R=[R;zeros(df.N-size(R,1),df.N)];
                X=[X;zeros(df.N-size(X,1),df.N)];
            end
            
            %%%%%%%%%%%% 电源处的出力 %%%%%%%%%%%%
            P_g=sdpvar(1,1);
            Q_g=sdpvar(1,1);
            %%%%%%%%%%%% 节点电压 %%%%%%%%%%%%
            U=sdpvar(df.N,1);
            
            PG=[P_g	zeros(1,117)];
            QG=[Q_g	zeros(1,117)];
            
            %%%约束项
            F=[];
            F=[F,0.93^2<=U<=1.07^2];
            F=[F,U(1)==1];

            F=[F,P_g<=5,P_g>=-5,Q_g<=5,Q_g>=-5];   %G related constraints
            F=[F, -1e3<=PL<=1e3,-1e3<=QL<=1e3,0<=IL<=1.5*(11/10)^2];   %,G related constraints  -5<=PL,-5<=QL,

            for i=1:df.N
                F=[F,(sum(PL(:,i)-R(:,i).*IL(:,i))+PG(i)-PN(i)-sum(PL(i,:)))==0,...  %  最后的sum改了正负号   +PDIS(i)-PCHAR(i)
                    (sum(QL(:,i)-X(:,i).*IL(:,i))+QG(i)-QN(i)-sum(QL(i,:)))==0];
            end
            for v=1:size(allb,1)
                if allb(v,1)<allb(v,2)
                    i=allb(v,1);j=allb(v,2);
                else
                    j=allb(v,1);i=allb(v,2);
                end
                F=[F,(U(i)-2*(R(i,j)*PL(i,j)+X(i,j)*QL(i,j))+(R(i,j)^2+X(i,j)^2)*IL(i,j)-U(j))==0];
                F=[F,norm([2*PL(i,j),2*QL(i,j),(U(i)-abs(IL(i,j)))])<=(U(i)+abs(IL(i,j)))];
                   %     F=[F,PL(i,j)^2+QL(i,j)^2>=U(i)*IL(i,j)];
            end
            
            g=(P_g.^2); 
            
            % optimization
            optimize(F,g,sdpsettings('solver','gurobi','gurobi.QCPDual',1));
            V=value(U);
            IL=[allb(:,1:2),value(I_l)];
            % IL=value(IL);
            Pg=value(P_g);
            QL = value(QL);
        end
        
        %% P2P计算
        function [ptr_matrix,FTR,g]=Calc_P2P(T, ptr_vec, df)
        %%%%% 输入
            % T: 所有边的状态，1为连接，0为断开, shape [1, -1]
            % ptr_vec:节点P2P交易量（列向量), shape [custom_agent_num, 1]
        %%%%% 输出
            % ptr_matrix: 第i行第j列代表i和j的交易量，第i行代表第i个主体,
                        % shape [custom_agent_num,custom_agent_num]
            % FTR: 第i项为第i给主体的P2P交易过网费成本
            % g: 所有主体总成本
            yalmip('clear')
            data = df.ans;
            data.branch(:,11)=T';
            YY=makeYbus(data); %生成节点导纳矩阵
            Y=full(YY);  %转化为矩阵形式
            BB=imag(Y);  %提取虚部，电抗
            allb=df.allbranch(find(T~=0),:);
            H=zeros(size(allb,1),df.N);
            for i=1:size(allb,1)
                H(i,allb(i,1))=-BB(allb(i,1),allb(i,2));
                H(i,allb(i,2))=+BB(allb(i,1),allb(i,2));
            end           
            
            PTDF=H*pinv(BB);
            % 初始化pt
            ptr_matrix = sdpvar(df.N, df.N);
            for i=1:df.N
                % pt中第i行第j列代表i和j的交易量，第i行代表第i个用户的交易分配
                ptr_matrix(i,:)=sdpvar(1,df.N);
            end
            
            F=[];
            F=[F,diag(ptr_matrix)==0,sum(ptr_matrix,2)==ptr_vec,triu(ptr_matrix,1)==-tril(ptr_matrix,-1)'];
            
            C = cell(df.N, df.N);
            for i=1:df.N
                C{i}=-1*eye(df.N);
                C{i}(i,:)=1;
                C{i}(:,i)=0;
                if ptr_vec(i)>=0
                F=[F,0<=ptr_matrix(i,:)<=ptr_vec(i)];
                else
                F=[F,0>=ptr_matrix(i,:)>=ptr_vec(i)];
                end
            end
            
            for i=1:df.N
                FTR(i)=df.gamma_wheel*(norm(PTDF*sum(C{i}.*repmat(ptr_matrix(i,:),...
                    df.N,1),2),1));
            end
            % 总成本
            g=(sum(FTR));
            % 优化过程
            optimize(F,g,sdpsettings('solver','gurobi'));
            % 提取sdpvar变量值
            ptr_matrix=value(ptr_matrix);
            g=value(g);
            FTR=value(FTR);
        end
        
    end
end