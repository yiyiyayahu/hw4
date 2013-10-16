function nb_test_err  = nb_learning(partition, num, X, Y, test_x, test_y)
    x_par = X(partition <= num,:);
    y_par = Y(partition <= num);
    nb = nb_train(x_par, y_par);
   
    nb_test_ret = nb_test(nb, test_x);
    nb_test_err = length(find(test_y ~= nb_test_ret))/length(test_y);
end
