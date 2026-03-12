import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.ndimage import gaussian_filter1d
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

class KineticDataLoader:
    def __init__(self, filepath, max_smiles_len=None, max_atoms=None):
        self.df = pd.read_csv(filepath)
        self.max_smiles_len = max_smiles_len
        self.max_atoms = max_atoms
        self.char_to_int = None
        self.vocab_size = None
        
        # 预处理器
        self.state_scaler = StandardScaler()
        self.beta_scaler = StandardScaler()
        self.tiso_scaler = StandardScaler()

    def preprocess_raw_data(self):
        """1. 数据平滑与基础清洗"""
        self.df['Rate_Smooth'] = np.nan
        group_keys = ['Molecule', 'Process_Param']
        
        for _, g in self.df.groupby(group_keys, sort=False):
            g = g.sort_values('Temp_K')
            sm = gaussian_filter1d(g['Rate_1_min'].to_numpy(dtype=float), sigma=1)
            self.df.loc[g.index, 'Rate_Smooth'] = sm
        
        # 转换为 ln(rate)
        self.df['ln_rate'] = np.log(np.clip(self.df['Rate_Smooth'], 1e-30, None))

    def get_molecular_features(self):
        """2. 提取 ECFP, SMILES 序列和 GNN 图特征"""
        unique_smiles = self.df['SMILES'].unique()
        
        # ECFP
        generator = GetMorganGenerator(radius=2, fpSize=2048)
        ecfp_dict = {}
        for s in unique_smiles:
            mol = Chem.MolFromSmiles(s)
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            ecfp_dict[s] = arr
            
        # SMILES Padding
        all_chars = sorted(set(''.join(unique_smiles)))
        self.char_to_int = {c: i + 1 for i, c in enumerate(all_chars)}
        self.vocab_size = len(self.char_to_int) + 1
        
        # GNN Utils
        def s_to_g(s):
            m = Chem.MolFromSmiles(s)
            nodes = np.array([[a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs()] 
                             for a in m.GetAtoms()], dtype='float32')
            adj = Chem.rdmolops.GetAdjacencyMatrix(m).astype('float32')
            return nodes, adj
        
        g_dict = {s: s_to_g(s) for s in unique_smiles}
        
        if not self.max_smiles_len:
            self.max_smiles_len = self.df['SMILES'].str.len().max()
        if not self.max_atoms:
            self.max_atoms = max(g[0].shape[0] for g in g_dict.values())
            
        return ecfp_dict, g_dict

    def prepare_tensors(self):
        """3. 将数据转换为模型输入张量"""
        self.preprocess_raw_data()
        ecfp_dict, g_dict = self.get_molecular_features()
        
        # 分子特征张量
        X_ecfp = np.array([ecfp_dict[s] for s in self.df['SMILES']]).astype('float32')
        X_smiles = pad_sequences([[self.char_to_int[char] for char in s] 
                                 for s in self.df['SMILES']], 
                                 maxlen=self.max_smiles_len, padding='post')
        
        X_nodes = np.zeros((len(self.df), self.max_atoms, 3))
        X_adjs = np.zeros((len(self.df), self.max_atoms, self.max_atoms))
        for i, s in enumerate(self.df['SMILES']):
            nf, adj = g_dict[s]
            X_nodes[i, :nf.shape[0], :] = nf
            X_adjs[i, :adj.shape[0], :adj.shape[0]] = adj

        # 物理/状态特征
        T_unscaled = self.df[['Temp_K']].values.astype('float32')
        alpha_unscaled = np.clip(self.df[['Alpha']].values, 1e-6, 1.0 - 1e-6).astype('float32')
        X_state_scaled = self.state_scaler.fit_transform(self.df[['Temp_K', 'Alpha']].values).astype('float32')
        
        # 模式特征 (Dynamic vs Isothermal)
        proc = self.df["Process_Param"].values
        is_dynamic = (proc <= 20.0).astype("float32").reshape(-1, 1)
        beta_feat = np.log(np.clip(np.where(is_dynamic.flatten()==1, proc, 1e-6), 1e-6, None)) * is_dynamic.flatten()
        tiso_feat = (1.0 / np.clip(np.where(is_dynamic.flatten()==0, proc+273.15, 1.0), 1.0, None)) * (1.0 - is_dynamic.flatten())
        
        beta_scaled = self.beta_scaler.fit_transform(beta_feat.reshape(-1, 1))
        tiso_scaled = self.tiso_scaler.fit_transform(tiso_feat.reshape(-1, 1))
        
        Y = self.df['ln_rate'].values.astype('float32')
        
        return {
            "inputs": [X_smiles, X_nodes, X_adjs, X_ecfp, X_state_scaled, 
                       T_unscaled, alpha_unscaled, is_dynamic, beta_scaled, tiso_scaled],
            "labels": Y
        }
