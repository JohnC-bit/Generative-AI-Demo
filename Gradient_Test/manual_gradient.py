import torch
import matplotlib.pyplot as plt 

def main():
    y = 1
    x1 = 2
    x2 = 3

    w1 = 0.11
    w2 = 0.21
    w3 = 0.12
    w4 = 0.08
    w5 = 0.14
    w6 = 0.15

    lr = 0.05

    losses = [] 
    counts = []
    sumw1 = []
    for i in range(30):

        h1 = w1*x1 + w2*x2 
        h2 = w3*x1 + w4*x2 

        y_hat = h1*w5 + h2*w6

        loss = (y-y_hat)**2/2

        d_Error_y_hat =y_hat-y
        d_y_hat_h1 = w5
        d_y_hat_h2 = w6

        d_h1_w1 = x1
        d_h1_w2 = x2
        d_h2_w3 = x1
        d_h2_w4 = x2

        w5 = w5 - lr*d_Error_y_hat*h1
        w6 = w6 - lr*d_Error_y_hat*h2

        w1 = w1 - lr*d_Error_y_hat*w5*x1
        w2 = w2 - lr*d_Error_y_hat*w5*x2
        w3 = w3 - lr*d_Error_y_hat*w6*x1
        w4 = w4 - lr*d_Error_y_hat*w6*x2

        if(i%3 == 0):
            print("Loss = ",loss)
            print("Y = ", y_hat)
            print("w1 = ", w1)
            losses.append(loss)
            sumw1.append(w1)
            counts.append(i)
        
    
    plt.plot(counts, sumw1, 'b-o')
    plt.title("W1 over counts")
    plt.show()

if __name__ == "__main__":
    main()