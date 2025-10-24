from src.pprl.encoder import HomomorphicEncryption
from src.pprl.matching import Matcher
import pandas as pd
from src.pprl.datamodel import Data
from openfhe import *
import re
import hashlib
import numpy as np

def Jaccard(overlap: int, lenght_a: int, lenght_b: int) -> float:
    return float ( overlap / (lenght_a + lenght_b - overlap))

def bigrams_to_uint64(df: pd.DataFrame): 
    
    def bigrams(row):
        salt = '#'
        bigrams = set()
        tokens = list(set(filter(None, re.split('[\\W_]', row.lower()))))

        for token in tokens:
            padded_token = salt + token  
            if len(padded_token) < 2: 
                padded_token += '#'
            for i in range(len(padded_token) - 2 + 1):
                bigram = str(padded_token[i:i+2])
                bigrams.add(bigram)

        ree_list = []
        for token in list(bigrams):
            ree = 0
            for idx, c in enumerate(reversed(token)):
                ree += (ord(c) & 0xff) << (8 * idx)
            ree_list.append(ree)            
        return ree_list
    
    
    
        #         # return re
                
        #         # int_bytes = int.from_bytes(hashed_bigram, 'big')
        #         # uint64 = np.uint64(int_bytes)
        #         # bigrams.add(uint64)



        # return list(bigrams)

    
    new_df = df.astype(str)
    new_df = new_df.map(lambda x: bigrams(x))
    return new_df
    
    
# NOTE: BLOCKING ME HOMOMORHIC THA EINAI DIAFORETIKOO


def matchViaOverlap(cc: CryptoContext, key_pair: KeyPair, plaintextA: Plaintext,
                    ciphertextB: Ciphertext, threshold: float, lengthB: int):
    
    a : list = plaintextA.GetPackedValue()

    if len(a) < lengthB: 
        while len(a) < lengthB: 
            a.append(-1)
    
    recA : Plaintext = cc.MakePackedPlaintext(a)
    overlap = 0.0
    
    d = cc.EvalSub(recA, ciphertextB)
    decrypt_res : Plaintext = cc.Decrypt(key_pair.secretKey, d)
    decrypt_res.SetLength(lengthB)
    
    vector_res : list = decrypt_res.GetPackedValue()
    
    
    overlap += sum(1 for v in vector_res if v == 0)

    length_a = len(a)
        
    for idx in range(1, length_a):
        a = a[1:] + a[:1]
        recA = cc.MakePackedPlaintext(a)
        d = cc.EvalSub(recA, ciphertextB)
        decrypt_res: Plaintext = cc.Decrypt(key_pair.secretKey, d)
        decrypt_res.SetLength(lengthB)
        vector_res : list = decrypt_res.GetPackedValue()
        
        overlap += sum(1 for v in vector_res if v == 0)
            
    jaccard = Jaccard(overlap, length_a, lengthB)


    return jaccard

    # if (a.() < b_size) {
    # while (a.size() < b_size) {
    #   a.push_back(INT_MIN);
    # }
#   }
    

if __name__ == '__main__':
    path = "data/test" 
    d1 = pd.read_csv(f"{path}/test1.csv")
    d2 = pd.read_csv(f"{path}/test2.csv")
    data = Data(dataset_1=d1, dataset_2=d2, id_column_name_1='id', id_column_name_2='id')
    
    he = HomomorphicEncryption()
    he_vectors = he.encode(data)
    
    print(he_vectors)
    
    he_matcher = Matcher(threshold=1.0)
    
    g = he_matcher.predict(data, he_vectors, type_of_match='extension')
    pairs_list = list()   
    for edge in g.edges:
        node1, node2 = edge
        if node1 < data.dataset_limit:
            id1 = data._gt_to_ids_reversed_1[node1]
            id2 = data._gt_to_ids_reversed_2[node2]
        
        else:
            id1 = data._gt_to_ids_reversed_1[node2]
            id2 = data._gt_to_ids_reversed_2[node1]

        pairs_list.append((id1, id2))

    
    pairs_df = pd.DataFrame(pairs_list, columns=['id1', 'id2'])
    
    print(pairs_df)
    
    exit(10)
    
    

    # d1 : pd.DataFrame = bigrams_to_uint64(data.dataset_1)
    # d2 : pd.DataFrame = bigrams_to_uint64(data.dataset_2)




    parameters = CCParamsBGVRNS()
    
    sigma = 3.2                       # palisade: sigma
    security_level = SecurityLevel.HEStd_128_classic    
    

    parameters.SetStandardDeviation(sigma)   # set sigma
    parameters.SetSecurityLevel(security_level)
    parameters.SetPlaintextModulus(65537)  # plaintext_modulus

    parameters.SetMultiplicativeDepth(1)   



    cc : CryptoContext = GenCryptoContext(parameters)


    cc.Enable(PKESchemeFeature.PKE)  # palisade encryption
    cc.Enable(PKESchemeFeature.LEVELEDSHE) #palisade she
    
    cc.Enable(PKESchemeFeature.ADVANCEDSHE) #palisade she
   
    
    # Generate a public/private key pair
    
                            # Palisade code
    key_pair : KeyPair = cc.KeyGen()  # LPKeyPair<DCRTPoly> keyPair;
                            # keyPair = cc->KeyGen();

    
    # Generate the relinearization key
    cc.EvalSumKeyGen(key_pair.secretKey)      # cc->EvalSumKeyGen(keyPair.secretKey);
    
    
    psi = 0
    threshold = 0.5
    
    for idx, row in d1.iterrows():
        for idx2, row2 in d2.iterrows():
            similarity = 0.0
            for attr in ['id', 'name', 'email', 'phone']: 
                plaintextA : Plaintext = cc.MakePackedPlaintext(row[attr])
                plaintextB : Plaintext = cc.MakePackedPlaintext(row2[attr])
                ciphertextB : Ciphertext = cc.Encrypt(key_pair.publicKey, plaintextB)
                similarity += matchViaOverlap(cc, key_pair, plaintextA, ciphertextB, threshold, plaintextB.GetLength())

            if similarity/4 >= 1.0:
                print(" ".join(data.dataset_1.iloc[idx].to_list()) , " ".join(data.dataset_2.iloc[idx].to_list()), similarity/4) 
                    
                
            
    

    # crypto_context.EvalMultKeyGen(key_pair.secretKey)
    
    # print(parameters)
    # print(cc)
    
    
    

    
    