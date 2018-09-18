#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:38:57 2018

@author: shamel
"""

import os
import numpy as np
import random
import csv
import warnings

# From RDkit import
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def read_smiles_file(filename, unique = True):
    
    smiles = []
    
    with open(filename,'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    f.close()
    
    if unique:
        smiles = list(set(smiles))  
    
    return smiles

def write_smiles_file(smiles, filename, unique = True):
    
    if unique:
        smiles = list(set(smiles))
        
    with open(filename,'w')as f:
        for smile in smiles:
            f.writelines((smile,'\n'))
    f.close()
    
    return

def canonize_smiles(smiles, sanitize = True):
    
    '''
    @param smiles: List of SMILES
    @param sanitize: Bool for specifying if we want to sanitize SMILES or not (http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol)
    @return canonized_smiles: List of canonical SMILES or ' ' if the SMILE string is invalid or cannot be sanitized 
    '''
    
    canonized_smiles = []
    for smile in smiles:
        try:
            canonized_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile,sanitize = sanitize)))
        except:
            warnings.warn(smile + ' Cannot be canonized: check SMILES string', UserWarning)
            canonized_smiles.append('')
    
    return canonized_smiles


def sanitize_smiles(smiles, canonize = True):
     
    '''
    @param smiles: List of SMILES
    @param canonize: Bool for specifying if we want to canonize SMILES or not (http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol)
    @return sanitized_smiles: List of sanitized SMILES or ' ' if the SMILE string is invalid or cannot be sanitized 
    '''
    
    sanitized_smiles = []
    for smile in smiles:
        try:
            if canonize:
                sanitized_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile,sanitize = True)))
            else:
                sanitized_smiles.append(smile)
        except:
            warnings.warn(smile + ' Cannot be sanitized: check SMILES string', UserWarning)
            sanitized_smiles.append('')
    
    return sanitized_smiles

def tokenize(smiles,add_default = True):
    
    # Common tokens found in SMILES
    default_tokens = [' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2',
                     '3', '4', '5', '6', '7', '8', '9', '<', '=', '>', '@', 'A',
                     'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
                     'c', 'e', 'i', 'l', 'n', 'o', 'p', 'r', 's']
    
    tokens = list(set(''.join(smiles)))
    
    if add_default:
        tokens = sorted(list(set(default_tokens+tokens)))
    else:
        tokens = sorted(list(set([' ']+tokens))) # Space here is used for padding SMILES to same length
    
    token2idx = dict((token,i) for i, token in enumerate(tokens))
    idx2token = dict((i,token) for i, token in enumerate(tokens))
    num_tokens = len(tokens)

    return tokens, token2idx, idx2token, num_tokens


def conv_smile_to_vec(smiles,max_len = 100):
    
    smile_vec = np.empty((len(smiles),max_len))
    smile_not_processed = []
    
    # Get the tokens
    tokens, token2idx, idx2token, num_tokens = tokenize(smiles)
    
    for i,smile in enumerate(smiles):
        if len(smile) < max_len:
            # Pad the SMILE with spaces
            smile = smile + ' '*(max_len-len(smile))
            for j,char in enumerate(smile):
                smile_vec[i,j] = token2idx[char]
        else:
            smile_not_processed.append(smile)
        
    return smile_vec, tokens, token2idx, idx2token, smile_not_processed
    
def get_fingerprint(smiles):
    
    array_fps = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        array_fps.append(arr)
    
    return array_fps


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    # Compute the dot product between u and v
    dot = np.dot(u,v)
    # Compute the L2 norm of u 
    norm_u = np.linalg.norm(u)
    
    # Compute the L2 norm of v 
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot/(norm_u*norm_v)
    
    return cosine_similarity
