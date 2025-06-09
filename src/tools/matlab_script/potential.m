
clear all

%% boring example for plot the correct landscape 

% def potential landscae
f = @(x) x - x.^3;
V = @(x) - (1.5.*(( x.^2 )./ 2) - (( x.^4)./4));
V = @(x) x.^2
x = linspace(-2, 2, 100);

%subplot(2, 1, 1);
plot(x, V(x)); title("Single equilibrium Dynamic");
xlabel('y');
ylabel('V(y)');


%subplot(2, 1, 2);
%plot(x, f(x)); title("f(x) = x dot")



%% Gaussian latent function potential 

syms x_g1 x_g2 x_g3 x;
% gaussian system
V_symbolic =  [
    -exp(-(x - x_g1).^2);
    -exp(-(x - x_g2).^2);
    -exp(-(x - x_g3).^2);
    ];

% defining real var 
y_t = linspace(-20, 20, 100);
g1 = 5; g2 = 1; g3 = 10;

% sum up for having a potential 
V = sum(V_symbolic);
% substitute
V = subs(V, x, y_t);
V = subs(V, [x_g1, x_g2, x_g3], [g1, g2, g3]);

%subplot(2,1,1)
plot(y_t, V); title("potential")

f = jacobian(sum(-V_symbolic), x);
f = subs(f, [x_g1, x_g2, x_g3], [g1, g2, g3]);
f = subs(f, x, y_t);
subplot(2, 1, 2);
plot(y_t, f); title("dynamic system") 



%% 3D plot --> new local minima : combination of the previous one
%y_g1 = 0;
%y_g2 = 14;
%y_g3 = 4;

r= 10;  % range 

%V = @(y1, y2) -exp(-(y1 - y_g1).^2) - exp(-(y2 - y_g1).^2) + ...
%              -exp(-(y1 - y_g2).^2) - exp(-(y2 - y_g2).^2) + ...
%              -exp(-(y1 - y_g3).^2) - exp(-(y2 - y_g3).^2); 

close()
[y1, y2] = meshgrid(linspace(-r, r, 100), linspace(-r, r, 100));
%subplot(2, 1, 1);

surf(v_multivar(y1, y2)); title('Multivariate Gaussian potential for 2D dynamics');
%subplot(2, 1, 2);
%contour(v_multivar(y1, y2))
%hold on

%x_point =[55.6, 45.4];
%y_point = [70, 31];
%attr = plot(x_point, y_point, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
%legend(attr, 'Attractors', 'Location', 'best');
%% 
hold off
%%
%% 3D plot --> new local minima : combination of the previous one
y_g1 = 0;
y_g2 = 14;
y_g3 = 4;

r= 10;  % range 

V = @(y1, y2) -exp(-(y1 - y_g1).^2) - exp(-(y2 - y_g1).^2) + ...
              -exp(-(y1 - y_g2).^2) - exp(-(y2 - y_g2).^2) + ...
              -exp(-(y1 - y_g3).^2) - exp(-(y2 - y_g3).^2); 


close()
[y1, y2] = meshgrid(linspace(-r, r, 100), linspace(-r, r, 100));
subplot(2, 1, 1);

surf(V(y1, y2)); title("Gaussian Potential for Single Attractor");
subplot(2, 1, 2);
contour(V(y1, y2))
hold on 
x_point =[50.4, 70, 50.4, 70 ];
y_point = [50.4, 50.4, 70, 70];
attr = plot(x_point, y_point, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
legend(attr, 'Attractors', 'Location', 'best');

%%
y_g1 = 5;
y_g2 = -5;

min = -20;
max = 20;
[y1, y2] = meshgrid(linspace(min, max, 100), linspace(min, max, 100));
V_2 = @(y1, y2)  ( ...
            (0.1*(y1 - y_g1).^2)./2 + ...
          + (0.1*(y2 - y_g2).^4)./4 - ( 2*((y2 - y_g2).^2)./2)    ...
          );

subplot(2, 1, 1);
surf(V_2(y1, y2));xlabel('y1'), ylabel('y2'), zlabel('Potential')

subplot(2, 1, 2);
contourf(V_2(y1, y2)); 




%% 3D state ---> volumetric plot
r = 10
[x, y, z ] = meshgrid(linspace(-r, r, 100), ...
    linspace(-r, r, 100), ...
    linspace(-r, r, 100));


alpha1 = 0.15;
alpha2 = 0.15;
alpha3 = 0.15;
gamma = r/2 + 1;

potential_3d = @(y1, y2, y3) + y1.^2 .* (alpha1./2) ...
                             + y2.^2 .* (alpha2./2) ...
                             + y3.^4 .* (alpha3./4) - gamma.*((y3.^2)./2);

P = potential_3d(x, y, z);
subplot(1, 1, 1)
slice(x, y, z, P, [-2, 2], [-2, 2], [-2, 2])
xlabel("y1")
ylabel("y2")
zlabel("y3")


%% PCA for reduce 300_dim to 3_dim
% still developing 
y = rand(300, 1);
y_g1 = ones(300, 1) * 8;
y_g2 = 4;
alpha = ones(300, 1) * 2;
gamma = 0.03;

negate_last = ones(300, 1); 
negate_last(length(negate_last)) = - negate_last(length(negate_last));

second_attractor = zeros(300, 1);
second_attractor(length(second_attractor)) = gamma.*((y(length(y)) - y_g2)^3);


f = @(y)  y.^2 .* (alpha./2).* negate_last + second_attractor;
pca_out = pca(f(y))
plot(f(y)); xlabel('y') f