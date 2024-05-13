x=sdpvar(3,1);
a=2*x(1)+3*x(2);
a=a+ones(1,3)*1e-300*x;
fliplr(coefficients(a,x)')
fliplr(coefficients(d(1,1)+ones(1,191)*1e-300*allcv(:,1),allcv(:,1))')

N_pi=N_pi+1; 
N_mu=N_mu+1;
N_cone=N_cone+1;
obj_pi=[obj_pi,xxxx];
obj_mu=[obj_mu,xxxx];
con_pi=[con_pi;xxxx];
con_mu=[con_mu;xxxx];
con_L=[con_L;xxxx];
con_L0=[con_L0;xxxx];

%%%%%%%%不等式
        N_pi(k)=N_pi(k)+1; 
        temp_con_pi=fliplr(coefficients(U(i,k)+ones(1,191)*1e-300*allcv(:,k),allcv(:,k))');
        temp_obj_pi=1-M*(1-R_real(i,k));
        eval(['con_pi',num2str(k),'=','[con_pi',num2str(k),';','temp_con_pi']);
        eval(['obj_pi',num2str(k),'=','[obj_pi',num2str(k),',','temp_obj_pi']);
%%%%%%%%等式
        N_mu(k)=N_mu(k)+1; 
        temp_con_mu=fliplr(coefficients(xxx+ones(1,191)*1e-300*allcv(:,k),allcv(:,k))');
        temp_obj_mu=xxx;
        eval(['con_mu',num2str(k),'=','[con_mu',num2str(k),';','temp_con_mu']);
        eval(['obj_mu',num2str(k),'=','[obj_mu',num2str(k),',','temp_obj_mu']);
%%%%%%%%二阶锥
        N_cone(k)=N_cone(k)+1;
        temp_con_L=[fliplr(coefficients(xxx+ones(1,191)*1e-300*allcv(:,k),allcv(:,k))');
                    fliplr(coefficients(xxx+ones(1,191)*1e-300*allcv(:,k),allcv(:,k))');
                    fliplr(coefficients(xxx+ones(1,191)*1e-300*allcv(:,k),allcv(:,k))')];
        temp_con_L0=flipud(coefficients(xxx+ones(1,191)*1e-300*allcv(:,k),allcv(:,k)));
        temp_obj_L0=0.05;
        eval(['con_L',num2str(k),'=','[con_L',num2str(k),';','temp_con_L']);
        eval(['con_L0',num2str(k),'=','[con_L0',num2str(k),',','temp_con_L0']);
        eval(['obj_L0',num2str(k),'=','[obj_L0',num2str(k),',','temp_obj_L0']);
        N_dual_cone(N_cone,k)=2;     
        
        
        
        
        
x=sdpvar(3,1);
y=sdpvar(3,1);
a=2*x.*y;
% b=3*x(2);
% a=[2 3 0]*x
coefficients(a,x.*y)


