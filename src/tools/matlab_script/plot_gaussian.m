
%f_0 = @(x)

f = @(x) ... 
-exp(-(x - a)^2) ...
-exp(-(x - b)^2) ...
-exp(-(x - c)^2);




df = @(x, y) ...
-exp(-(x - a)^2)*(-(x - a)^2)*(-a) ...
-exp(-(x - b)^2)*(-(x - b)^2)*(-b) ...
-exp(-(x - c)^2)*(-(x - c)^2)*(-c);
fsurf(f, df)
shading interp
axis tight
%%
syms x
y = [-500:+500];
% f = -exp(-(x+1)^2) -exp(-(x - 1)^2);
f = - ((x.^2)./2 - (x.^4)./4)
V_x = int(f);
plot(y, subs(V_x, x, y))