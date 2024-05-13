case33bw;
b=ans.branch(:,1:2);
g=graph(b(:,1),b(:,2),ones(37,1));
B=adjacency(g);
B=full(B);
D=distances(g);