function ret = cd1(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_state = sample_bernoulli(visible_state); %when real world data
    
    % 1 Calculate hidden weight probabilities using defined visible units
    % 2 Calculate hidden states using bernoulli
    % 3 Use hidden states to then calculate probabilities of the next visible states
    % 4 Sample the probabilities of the visible states to get the new visible states (again using Bernoulli
    % 5 Use the new visible states to calculate probability of the hidden states a second time
    % 6 Sample the probabilities to obtain new hidden states
    % 7 Calculate the goodness gradient for the initial visible states and hidden states
    % 8 Calculate the goodness gradient for the final visible and hidden states
    % 9 subtract 8 from 7 and return this matrix
    
    hidden_state = sample_bernoulli(logistic(rbm_w * visible_state)); % 2
    reconstruction_state = sample_bernoulli(logistic(rbm_w' * hidden_state)); % 4
    %hidden_state2 = sample_bernoulli(logistic(rbm_w * reconstruction_state)); % 6
    hidden_state2 = logistic(rbm_w * reconstruction_state); % 6-modified
    
    ggradient = hidden_state * visible_state';
    ggradient2 = hidden_state2 * reconstruction_state';

    ret = (ggradient - ggradient2) ./ size(visible_state, 2);

end