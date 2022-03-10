clearvars; close all
set(0,'DefaultFigureWindowStyle','docked')
step = 200;
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;
V = linspace(-1.95,0.7,step);
I = Is*exp((1.2*V/0.025)-1)+ (Gp*V) - Ib*exp(((-1.2/0.025)*(V+Vb))-1);
I2 = randn(1,200) .*(0.2*I)+I;

%using polyfit and polyval 
polyI4 = polyfit(V,I,4);
fitI4 = polyval(polyI4,V,4);

polyI8 = polyfit(V,I,8);
fitI8 = polyval(polyI4,V,8);

polyI24 = polyfit(V,I2,4);
fitI24 = polyval(polyI24,V,4);

polyI28 = polyfit(V,I2,8);
fitI28 = polyval(polyI28,V,8);


inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs


% %plot using fit
% x = linspace(-1.95,0.7,step)';
% I_T = I2';
% fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
% ff = fit(x,I_T,fo);
% If = ff(x);
% plot(If)

%plot Data 
subplot(2,3,1)
plot(V,I)
hold on
plot(V,fitI4,'Color', 'g')
hold on
plot(V,fitI8,'Color', 'b')
legend('data','4th order','8th order')
title('Data vs Polyfit')
subplot(2,3,2)
plot(V,I2)
hold on
plot(V,fitI24,'Color', 'y')
hold on
plot(V,fitI28,'Color', 'r')
legend('data','4th order','8th order')
title('Data vs Polyfit of I2')
%plotting log values
subplot(2,3,3)
semilogy(V,abs(I))
title('log I')
subplot(2,3,4)
semilogy(V,abs(I2))
title('log I2')
%plotting neural net
subplot(2,3,5)
plot(V,Inn)
title('neural network')
subplot(2,3,6)
semilogy(V,abs(Inn))
title('log neural network')

