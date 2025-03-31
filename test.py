from hdf5storage import savemat

from utils_torch1 import *
from utils_LLM import MyGPT2Model
Nt = 256  # the number of antennas
P = 1  # the normalized transmit power

def pred(method):
    if method == 'ODELSTM':
        rate = []
        pathTest = 'train_set/test/'+str(PNR)+'dB'
        H1, H_est1 = mat_load_test(pathTest)
        H_input1 = np.expand_dims(H_est1, axis=1)
        H_input1 = np.concatenate([np.real(H_input1), np.imag(H_input1)], axis=1)
        # H_input1 = H_input1.reshape(H_input1.shape[0], -1)
        # H_input1 = np.concatenate([np.real(H_est1), np.imag(H_est1)], axis=1)
        H1 = np.squeeze(H1)
    # model = MyModel(H_input1, Nt)
        model = Model_ODELSTM(in_features=256,
                              hidden_size=256,
                              out_feature=256,
                              return_sequences=True,
                              # solver_type="fixed_rk4"
                              solver_type="fixed_euler")
        # Load the trained model weights
        # model.load_state_dict(torch.load('./saved_models/20dB.pth'))
        model.load_state_dict(torch.load('saved_models/'+str(PNR)+'dB_ODELSTM.pth'))
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for snr in range(20, 21, 2):
                SNR1 = np.power(10, np.ones([H1.shape[0], 1]) * snr / 10)
                # y = model(torch.tensor(H_input1, dtype=torch.float32),
                #           torch.tensor(H1, dtype=torch.complex64),
                #           torch.tensor(SNR1, dtype=torch.float32), Nt).mean()
                y, vf = model(torch.tensor(H_input1, dtype=torch.float32), indim, None, outdim,
                          torch.tensor(H1, dtype=torch.complex64),
                          torch.tensor(SNR1, dtype=torch.float32), Nt)
                rate.append(-y.mean())
                print(snr, y.mean())

        # print(rate)
        # rate = np.array(rate)
        # vf_numpy = vf.detach().cpu().numpy()
        # savemat('precoder/10dB/precoderv' + str(v) + method + '.mat', {'vf': vf_numpy})
    elif method == 'BFNN':
        rate = []
        pathTest = 'train_set/test/'+str(PNR)+'dB'
        H1, H_est1 = mat_load_test(pathTest)
        H_input1 = np.concatenate([np.real(H_est1), np.imag(H_est1)], axis=2)
        H1 = np.squeeze(H1)
        SNR1 = np.power(10, np.random.randint(20, 21, [H1.shape[0], 1]) / 10)
        # H_input1 = np.concatenate([np.real(H_est1), np.imag(H_est1)], axis=2)
        H1 = np.squeeze(H1)
        model = MyModel(H_input1, Nt)
        model.load_state_dict(torch.load('saved_models/'+str(PNR)+'dB_'+method+'.pth'), False)
        model.eval()
        with torch.no_grad():
            for snr in range(20, 21, 2):
                SNR1 = np.power(10, np.ones([H1.shape[0], 1]) * snr / 10)
                y, vf = model(torch.tensor(H_input1, dtype=torch.float32),
                          torch.tensor(H1, dtype=torch.complex64),
                          torch.tensor(SNR1, dtype=torch.float32), Nt)
                rate.append(-y.mean())
                print(snr, y.mean())
    elif method == 'LLM':
        model = MyGPT2Model()
        # Load the trained model weights
        # model.load_state_dict(torch.load('./saved_models/20dB.pth'))
        model.load_state_dict(torch.load('saved_models/' + str(PNR) + 'dB_LLM_MLP1.pth'))
        model.eval()
        rate = []
        pathTest = 'train_set/test/'+str(PNR)+'dB'
        H1, H_est1 = mat_load_test(pathTest)
        # H_input1 = np.concatenate([np.real(H_est1), np.imag(H_est1)], axis=2)  # (batch_size, 2, 255)
        # H_input1 = H_input.reshape(H_input1.shape[0], H_input1.shape[1], -1)  # (batch_size, time, 2*Nt)
        H_input1 = np.expand_dims(H_est1, axis=1)
        H_input1 = np.concatenate([np.real(H_input1), np.imag(H_input1)], axis=1)

        H_input1 = torch.tensor(H_input1, dtype=torch.float32)
        H1 = torch.tensor(H1, dtype=torch.complex64)

        model.eval()
        with torch.no_grad():
            for snr in range(20, 21, 2):
                SNR1 = np.power(10, np.ones([H1.shape[0], outdim, 1]) * snr / 10)
                SNR1 = torch.tensor(SNR1, dtype=torch.float32)

                y, vf = model(H_input1, H1, SNR1)
                rate.append(-y.mean())
                print(f'SNR: {snr}, Output: {y.mean()}')
        # print(rate)
        # vf_numpy = vf.detach().cpu().numpy()
        # savemat('precoder/10dB/precoderv' + str(v) + method + '.mat', {'vf': vf_numpy})
        # rate = np.array(rate)
    print(rate)
    vf_numpy = vf.detach().cpu().numpy()
    savemat('precoder/' + str(PNR) + 'dB/precoderv' + str(v) + method + '.mat', {'vf': vf_numpy})
    vff = np.array(vf.detach().cpu().numpy())
    return vff

methods = ['ODELSTM']
for method in methods:
    pred(method)