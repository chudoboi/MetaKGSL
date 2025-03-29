import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#plt.rcParams['font.sans-serif']= ['Heiti TC'] # set up the font
plt.rcParams['axes.unicode_minus']=False # use ASCII hyphen to display tick labels at negative values

def show_loss(): 
   # read Loss.txt file
    with open('Loss.txt', 'r') as file:
        content = file.read()
    
    # split the content of the file into two parts (encoder and decoder)
    loss_parts = content.split('==='*20)  
    encoder_loss_str, decoder_loss_str = loss_parts[0].strip(), loss_parts[1].strip()
    
    # use regular expressions to extract the average loss value in each row
    encoder_list_loss = [float(re.search(r'average loss ([0-9]*\.?[0-9]+)', line).group(1)) 
                         for line in encoder_loss_str.split('\n') 
                         if re.search(r'average loss ([0-9]*\.?[0-9]+)', line)]
    
    decoder_list_loss = [float(re.search(r'average loss ([0-9\.]+)', line).group(1)) 
                         for line in decoder_loss_str.split('\n') 
                         if re.search(r'average loss ([0-9\.]+)', line)]
    
    # create a side-by-side line chart
    epochs = list(range(20))  # Suppose there are 20 epochs with x-axis values from 0 to 19
    
    # create 2 subgrams and arrange them horizontally
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4),dpi=300)
    
    # draw the loss curve of the encoder
    ax1.plot(epochs, encoder_list_loss, label='Encoder Loss', color='blue', marker='o')
    ax1.set_title('Encoder Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True)
    ax1.legend()
    
    # draw the loss curve of the decoder
    ax2.plot(epochs, decoder_list_loss, label='Decoder Loss', color='red', marker='x')
    ax2.set_title('Decoder Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Value')
    ax2.grid(True)
    ax2.legend()
    
    # set the x-axis scale to an integer, ranging from 0 to 19
    ax1.set_xticks(range(20))
    ax2.set_xticks(range(20))
    

    plt.xticks(range(20))
    plt.tight_layout()
    
    plt.savefig('./Loss_curve.jpg')
    plt.show()

def show_hits():
    
    def get_dict(temp_epoch_datas):
        result = []
        a_cleaned = re.sub(r'^[=\n]+', '', temp_epoch_datas).split('\n') 
        temp_dict = {}
        for s in a_cleaned:
            for key in keys:
                if key in s:
                    try:
                        floats = re.findall(r'\d+\.\d+', s)
                        temp_dict[key] = floats[0]
                    except:
                        continue
                    break
        result.append(temp_dict)
        return result
    
    with open('result.txt', 'r') as file:
        data = file.read()
    
    epoch_data = re.split(r'={20}(\d+)=+', data)
    
    result1, result2 = [], []
    keys = ['Hits@100','Hits@10','Hits@3','Hits@1','Mean rank','Mean Reciprocal Rank']
    # traverse each epoch
    for epoch in range(2,len(epoch_data),2):  # epoch_data[0] is an empty string
        temp_epoch_data = epoch_data[epoch].split('\n=')
        result1.extend(get_dict(temp_epoch_data[0]))
        result2.extend(get_dict(temp_epoch_data[1]))
    
    df1 = pd.DataFrame(result1)
    df1.to_excel('result1.xlsx')
    df2 = pd.DataFrame(result2)
    df2.to_excel('result2.xlsx')
                
def show_result():
    df = pd.read_excel('./result/result1.xlsx')
    
   
    
    # get all column names (except 'epoch')
    columns = [col for col in df.columns if col != 'epoch']
    # set color mapping
    colors = cm.get_cmap('tab10', len(columns))
    
    # draw a separate line chart for each column
    for idx, column in enumerate(columns):
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df[column], label=column, color=colors(idx))
        
        plt.xlabel('Epoch')
        plt.ylabel(f'{column}')
        
        plt.legend()
        plt.savefig('./result/' + column +'.jpg')
        plt.show()
    
if __name__ == "__main__":
    show_result()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
