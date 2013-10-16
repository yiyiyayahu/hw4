function lr_test_err = lr_learning(partition, num, X, Y, test_x, test_y)
    varargin.step_size = 1e-3;
    varargin.stop_tol = 1e-5;
    varargin.max_iter = 1000;
    C = 1e-3;
    
    x_par = X(partition <= num,:);
    y_par = Y(partition <= num);
    
    [w,~,~] = lr_train(x_par, y_par, C, varargin);

    lr_test_ret = lr_test(w, test_x);
    lr_test_err = length(find(test_y ~= lr_test_ret))/length(test_y);
end
