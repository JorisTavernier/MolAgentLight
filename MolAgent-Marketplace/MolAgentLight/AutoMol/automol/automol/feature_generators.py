"""implementation of the different feature generators and their base class.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025.
"""

# from .model import *  # Removed: model.py no longer exists (PyTorch dependency removed)

import json
import re

import pandas as pd

import numpy as np
from pathlib import Path
from rdkit import __version__ as rdkit_version
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors


from importlib_resources import files


#: Canonical encoder feature keys, in listing order. ``Bottleneck`` is the default.
CANONICAL_ENCODER_KEYS = (
    "Bottleneck",
    "Bottleneck_chembl37_base",
    "Bottleneck_chembl27",
)

#: Extra keys accepted as input that resolve to a canonical key. Aliases share
#: the canonical key's instance, so they cost no additional ONNX session.
FEATURE_KEY_ALIASES = {
    "Bottleneck_chembl37_logd": "Bottleneck",
}


def _package_relative(path):
    """Return ``path`` relative to the automol package dir, or None if outside it."""
    if path is None:
        return None
    base = Path(os.path.dirname(os.path.realpath(__file__)))
    try:
        return Path(path).resolve().relative_to(base).as_posix()
    except (ValueError, OSError):
        return None


def _resolve_package_path(rel, absolute):
    """Prefer the package-relative location; fall back to the stored absolute path.

    Lets a model trained under one install path load under another, while
    keeping older pickles that recorded only an absolute path working.
    """
    if rel is not None:
        candidate = Path(os.path.dirname(os.path.realpath(__file__))) / rel
        if candidate.exists():
            return str(candidate)
    return absolute


def default_encoder():
    """Return a fresh instance of the default encoder (v6_best, E-logD, ChEMBL 37)."""
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return MolBottleGenerator(
        export_dir=os.path.join(base_dir, "encoders", "e_logd"),
        variant="e_logd",
    )


def retrieve_default_offline_generators(model='CHEMBL', radius=2, nbits=2048):
    """
    Function that returns a dictionary of default internal feature generators.

    ``Bottleneck`` is the default encoder: v6_best (E-logD, full ChEMBL 37,
    epoch 39). ``Bottleneck_chembl37_base`` is E-base — the only encoder in the
    set without logD supervision, and therefore the one to use for logD, logP
    and lipophilicity endpoints. ``Bottleneck_chembl27`` is the legacy incumbent.

    Args:
        model: string that to define which kind of Deeplearning generated features. Reflects the smiles used for training the encoder.
        radius: radius of ecfp generation
        nbits: size of the ecfp features
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))

    logd = default_encoder()
    generators = {
        'Bottleneck': logd,
        'Bottleneck_chembl37_base': MolBottleGenerator(
            export_dir=os.path.join(base_dir, "encoders", "e_base"),
            variant="e_base",
        ),
        'Bottleneck_chembl27': OnnxBottleneckTransformer(),
        'rdkit': RDKITGenerator(),
        f'fps_{nbits}_{radius}': ECFPGenerator(radius=radius, nBits=nbits),
    }
    for alias, canonical in FEATURE_KEY_ALIASES.items():
        generators[alias] = generators[canonical]
    return generators

###############################
class FeatureGenerator():
    def __init__(self):
        """
        Initialization of the base class
        """
        ## number of features
        self.nb_features=-1
        ## list of the names of the features
        self.names=[]
        ## the name of the generator
        self.generator_name=''
    
    def get_nb_features(self):
        """
        getter for the number of features.
        
        includes an assert that number of features is positive.
        
        Returns
            nb_features (int): number of features
        """
        assert self.nb_features>0, 'method not correctly created, negative number of features'
        return self.nb_features
    
    def check_consistency(self):
        """
        checks if the number of features is positive and the length of the feature names equal the number of features
        """
        assert len(self.names)==self.nb_features, 'Provided number of names is not equal to provided number of features'
        assert self.nb_features>0, 'negative number of features'
    
    def generate(self,smiles):
        """
        generate the feature matrix from a given list of smiles
        
        Args:
            smiles: list of smiles (list of strings)
        
        Returns:
            X: feature matrix as numpy array 
        """
        pass
    
    def generate_w_pairs(self,smiles,original_indices,new_indices):
        """
        generate the feature matrix from a given list of smiles
        
        Args:
            smiles: list of smiles (list of strings)
            original_indices: indices for pairs of ligands without reindexing after datasplitting
            new_indices: list indices for pairs of ligands with reindexing after datasplitting
        Returns:
            X: feature matrix as numpy array 
        """
        X=self.generate(smiles)
        X_p=np.zeros((len(new_indices),2*X.shape[1]))           
        for idx,(i,j) in enumerate(new_indices):
            X_p[idx,:]=np.hstack((X[i,:],X[j,:]))
        return X_p
    
    def get_names(self):
        """
        getter for the names of the features
        
        Returns:
            names (List[str]): list of names
        """
        return self.names
    
    def get_generator_name(self):
        """
        getter for the generator name
        
        Returns:
            generator_name (str): the name of the generator
        """
        return self.generator_name

###############################    
class RDKITGenerator(FeatureGenerator):
    """
    feature generator returning the rdkit descriptors
    """

    def __init__(self):
        """
        Initialization 
        """
        ## list of rdkit names from rdkit.Chem.Descriptors.descList
        self.rdkitnames=[ n for n,f in Descriptors.descList]
        ## descriptor calculator MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkitnames)
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkitnames)
        ## list of names of the features
        self.names= self.calculator.GetDescriptorNames()
        ## number of features
        self.nb_features=len(self.rdkitnames)
        ## generator name
        self.generator_name=f'automol_rdkit_{rdkit_version}'
        
    def get_descriptor(self,s):
        """
        retrieve rdkit descriptors for given smiles s
        
        return tuple of nans if the rdkit fails to calculate descriptors
        
        Args:
            s (str): smiles string
        
        Returns:
            rdkit descriptors or nans
        """
        if s=="" or s is None:
            return self.nb_features*(np.nan,)
        try:
            m=MolFromSmiles(s)
            if m :
                return self.calculator.CalcDescriptors(m)
            else:
                return  self.nb_features*(np.nan,)
        except:
            return self.nb_features*(np.nan,)
        
    def generate(self,smiles):
        """
        Generate all given descriptors for given list of smiles and return as numpy array
        
        Args:
            smiles: list of smiles (list of strings)
        
        Returns:
            des: feature matrix as numpy array 
        """
        des = np.array([self.get_descriptor(x) for x in smiles])
        #test nan 
        #des[-1]=np.array(self.nb_features*(np.nan,))
        return des
    
###############################
class ECFPGenerator(FeatureGenerator):
    """
    The chemical fingerprints generator using rdkit
    """

    def __init__(self,radius=2, nBits =2048,useChirality= False,useFeatures= False):
        """
        Initialization of the ecfp generator
        
        see rdkit.AllChem.GetMorganFingerprintAsBitVect for details on the morgan fingerprint generation
        
        Args:
            radius: radius for morgan fingerprints [=2]
            nBits: number of bits used [=2048]
            useChirality: boolean to set to use chirality when computing fps[=False]
            useFeatures: boolean to set [=False]
        """
        ## radius
        self.radius=int(radius)
        ## nbits
        self.nBits=int(nBits)
        ## boolean to indicated use of chirality when computing fps
        self.useChirality=useChirality
        ## boolean to indicated use of features when computing fps
        self.useFeatures=useFeatures
        ## number of features
        self.nb_features=int(nBits)
        ## list of feature names
        self.names=[f'fps_{i}_of_{nBits}_radius_{radius}' for i in range(int(nBits))]
        ## generator name
        self.generator_name=f'automol_ecfp_{nBits}_radius_{radius}_rdkit_{rdkit_version}'
    
    def generate(self,smiles):
        """
        Generate ecfp for given list of smiles and return as numpy array
        
        Args:
            smiles: list of smiles (list of strings)
        
        Returns:
            X: feature matrix as numpy array
        """
        #mols =[Chem.MolFromSmiles(s) for s in smiles]
        
        return np.array([np.array(fps) for fps in self.generate_BitVect(smiles)],dtype=float)
        #test nan 
        #outputs=np.array([np.array(fps) for fps in self.generate_BitVect(smiles)],dtype=float)
        #outputs[-1]=np.array(self.nb_features*(0,))
        #return outputs
    
    def generate_BitVect(self,smiles):
        """generate the features as BitVects 
        
        Args:
            smiles (list[str]): list of smiles (list of strings)
        
        Returns:
            The list of bitVect belonging to the given smiles
        """
        def getFP(s):
            try:
                return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=self.radius,
                                                     nBits=self.nBits,
                                                     useChirality=self.useChirality,useFeatures=self.useFeatures)
            except:
                return self.nBits*[np.nan]
        return [getFP(s) for s in smiles]
                    
class MolfeatGenerator(FeatureGenerator):
    def __init__(self):
        super().__init__()
    
    def generate(self, smiles):
        self.check_consistency()
        st=0
        end=0
        X_list=[]
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            smiles_l=smiles[st:end]
            features = np.full([len(smiles_l), self.nb_features], np.nan)
            indices = []
            structures = []
            for i, s in enumerate(smiles_l):
                if s is None or s=='':
                    continue
                try:
                    m=Chem.MolFromSmiles(s)
                except Exception:
                    continue
                if m is not None:
                    indices.append(i)
                    structures.append(Chem.MolToSmiles(m))

            if structures:
                features[indices] = np.stack(self.model(structures))
            
            st+=self.batch_size
            X_list.append(features)
        return np.concatenate(X_list, axis=0)

class MolfeatPretrainedHFTransformer(MolfeatGenerator):
    def __init__(self, kind='MolT5', notation='smiles', dtype=float,max_length=220,batch_size=250):
        from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
        
        super().__init__()
        self.model = PretrainedHFTransformer(kind=kind, notation=notation, dtype=dtype,max_length=max_length)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_PretrainedHFTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size


class MolfeatFPVecTransformer(MolfeatGenerator):
    def __init__(self, kind='desc2D', dtype=float,batch_size=250):
        from molfeat.trans.fp import FPVecTransformer
        
        super().__init__()
        self.model = FPVecTransformer(kind=kind, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_FPVecTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size
        
class Molfeat3DFPVecTransformer(MolfeatGenerator):
    def __init__(self, kind='desc2D', dtype=float,batch_size=250,seed=42):
        from molfeat.trans.fp import FPVecTransformer
        
        super().__init__()
        self.model = FPVecTransformer(kind=kind, dtype=dtype)
        self._seed=seed
        
        m = Chem.MolFromSmiles('Oc1ccc(cc1OC)C=O')
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=self._seed)
        X_try=np.stack(self.model([m]))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_FPVecTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size
        
    def generate(self, smiles):
        self.check_consistency()
        st=0
        end=0
        X_list=[]
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            smiles_l=smiles[st:end]
            features = np.full([len(smiles_l), self.nb_features], np.nan)
            indices = []
            structures = []
            for i, s in enumerate(smiles_l):
                if s is None or s=='':
                    continue
                try:
                    m = Chem.MolFromSmiles(s)  # talidomide
                    m = Chem.AddHs(m)
                    AllChem.EmbedMolecule(m, randomSeed=self._seed)
                except Exception:
                    continue
                if m is not None:
                    indices.append(i)
                    structures.append(m)

            if structures:
                features[indices] = np.stack(self.model(structures))

            st+=self.batch_size
            X_list.append(features)
        return np.concatenate(X_list, axis=0)


class MolfeatMoleculeTransformer(MolfeatGenerator):
    def __init__(self, featurizer='mordred', dtype=float,batch_size=250):
        from molfeat.trans import MoleculeTransformer
        
        super().__init__()
        self.model = MoleculeTransformer(featurizer=featurizer, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        if isinstance(featurizer,str):
            self.generator_name = f'automol_MoleculeTransformer_{featurizer}'
        else:
            self.generator_name = f'automol_MoleculeTransformer'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size
        
class Molfeat3DMoleculeTransformer(MolfeatGenerator):
    def __init__(self, featurizer='mordred', dtype=float,batch_size=250,seed=42):
        from molfeat.trans import MoleculeTransformer
        
        super().__init__()
        self._seed=seed
        self.model = MoleculeTransformer(featurizer=featurizer, dtype=dtype)
        
        m = Chem.MolFromSmiles('Oc1ccc(cc1OC)C=O')
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=self._seed)
        X_try=np.stack(self.model([m]))
        self.nb_features=X_try.shape[1]
        if isinstance(featurizer,str):
            self.generator_name = f'automol_3dMoleculeTransformer_{featurizer}'
        else:
            self.generator_name = f'automol_3dMoleculeTransformer'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size        
        
    def generate(self, smiles):
        self.check_consistency()
        st=0
        end=0
        X_list=[]
        while end < (len(smiles)):
            end= min(st+self.batch_size, len(smiles))
            smiles_l=smiles[st:end]
            features = np.full([len(smiles_l), self.nb_features], np.nan)
            indices = []
            structures = []
            for i, s in enumerate(smiles_l):
                if s is None or s=='':
                    continue
                try:
                    m = Chem.MolFromSmiles(s)  # talidomide
                    m = Chem.AddHs(m)
                    AllChem.EmbedMolecule(m, randomSeed=self._seed)
                except Exception:
                    continue
                if m is not None:
                    indices.append(i)
                    structures.append(m)

            if structures:
                features[indices] = np.stack(self.model(structures))
            
            st+=self.batch_size
            X_list.append(features)
        return np.concatenate(X_list, axis=0)
    
        

class MolfeatPretrainedDGLTransformer(MolfeatGenerator):
    def __init__(self, kind='gin_supervised_edgepred', dtype=float,batch_size=250):
        from molfeat.trans.pretrained import PretrainedDGLTransformer
        
        super().__init__()
        self.model =  PretrainedDGLTransformer(kind=kind, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_MPretrainedDGLTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size

    
class MolfeatGraphormerTransformer(MolfeatGenerator):
    def __init__(self, kind='pcqm4mv2_graphormer_base', dtype=float,batch_size=250):
        from molfeat.trans.pretrained import GraphormerTransformer
        
        super().__init__()
        self.model =  GraphormerTransformer(kind=kind, dtype=dtype)
        
        X_try=np.stack(self.model(['Oc1ccc(cc1OC)C=O']))
        self.nb_features=X_try.shape[1]
        self.generator_name = f'automol_GraphormerTransformer_{kind}'
        self.names.extend(f'feature_{x}' for x in range(self.nb_features))
        self.batch_size=batch_size



from pathlib import Path
import os
from typing import List, Optional, Union

import numpy as np

from .tokenization import Vocabulary, SmilesTokenizer


class OnnxBottleneckTransformer(FeatureGenerator):
    """
    ONNX Runtime version of BottleneckTransformer.

    Generates 250-dimensional features from SMILES strings using a
    pre-trained transformer encoder exported to ONNX format.

    No PyTorch required for inference.

    Attributes:
        n_features: Number of features (250)
        names: Feature names
        session: ONNX Runtime inference session
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        batch_size: int = 100,
        seq_len: int = 220,
    ):
        """
        Initialize the ONNX-based bottleneck transformer.

        Args:
            model_path: Path to ONNX model file. If None, uses default.
            vocab_path: Path to vocabulary JSON file. If None, uses default.
            providers: ONNX Runtime execution providers. Default: ['CPUExecutionProvider']
            batch_size: Batch size for processing (for memory efficiency)
            seq_len: Maximum sequence length for SMILES
        """
        super().__init__()

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime is required for inference. "
                "Install with: pip install onnxruntime"
            )

        # Default paths
        base_dir = os.path.dirname(os.path.realpath(__file__))

        if model_path is None:
            model_path = base_dir + '/bottleneck_encoder.onnx'
        if vocab_path is None:
            vocab_path = base_dir + '/vocab.json'

        self.model_path = model_path
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Initialize tokenizer with vocabulary
        self.tokenizer = SmilesTokenizer(
            vocab_path=vocab_path,
            max_seq_len=seq_len,
            add_sos=True,
            add_eos=True,
        )

        # Create ONNX Runtime session
        providers = providers or ['CPUExecutionProvider']

        # Check if model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run the conversion script first: "
                "python -m automol_onnx.conversion.export_bottleneck"
            )

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output names from model
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Feature configuration
        self.nb_features = 250
        self.names = [f'Bottleneck_{i}_of_{self.nb_features}_model_CHEMBL' for i in range(self.nb_features)]
        self.generator_name = f'automol_onnx_bn_{Path(model_path).stem}'

    def _run_inference(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on tokenized input.

        Args:
            input_ids: Token array of shape [seq_len, batch_size]

        Returns:
            Feature array of shape [batch_size, 250]
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_ids}
        )
        return outputs[0]

    def generate(self, smiles: Union[List[str], str]) -> np.ndarray:
        """
        Generate features for SMILES strings.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            numpy array of shape (n_samples, 250)
        """
        # Handle single SMILES
        if isinstance(smiles, str):
            smiles = [smiles]

        # Handle pandas Series/DataFrame
        if hasattr(smiles, 'tolist'):
            smiles = smiles.tolist()

        # Process in batches for memory efficiency
        all_features = []

        for start_idx in range(0, len(smiles), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(smiles))
            batch = smiles[start_idx:end_idx]

            # Tokenize batch: [seq_len, batch_size]
            input_ids = self.tokenizer.tokenize_batch(batch)

            # Run inference: [batch_size, 250]
            features = self._run_inference(input_ids)

            all_features.append(features)

        # Concatenate all batches
        return np.vstack(all_features)

    def __call__(
        self,
        smiles: Union[List[str], str],
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate features (callable interface).

        Args:
            smiles: Single SMILES or list of SMILES
            batch_size: Override batch size
            seq_len: Override sequence length (not used, kept for API compatibility)

        Returns:
            Feature array
        """
        if batch_size is not None:
            self.batch_size = batch_size

        return self.generate(smiles)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.names

    @property
    def n_features(self) -> int:
        """Number of features generated."""
        return self.nb_features

    def __repr__(self) -> str:
        return (
            f"OnnxBottleneckTransformer("
            f"model='{Path(self.model_path).name}', "
            f"nb_features={self.nb_features})"
        )

    def __getstate__(self):
        """
        Get state for pickling. Exclude the unpicklable ONNX session.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable ONNX session
        state['session'] = None
        state['input_name'] = None
        state['output_name'] = None
        state['_model_path_rel'] = _package_relative(self.model_path)
        return state

    def __setstate__(self, state):
        """
        Set state when unpickling. Recreate the ONNX session.
        """
        self.__dict__.update(state)
        # Recreate the ONNX session
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime is required for inference. "
                "Install with: pip install onnxruntime"
            )
        self.model_path = _resolve_package_path(state.get('_model_path_rel'), state['model_path'])
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name


###############################
# MolBottle tokenization — verbatim from molbottle.data.tokenizer.SMILES_PATTERN so
# this class stays self-contained (no molbottle install required in the consumer).
_MOLBOTTLE_SMILES_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|Al|As|Ag|Au|Be|Ba|Bi|Ca|Cu|Fe|Kr|He|Li|Mg|Mn|Na|Ni"
    r"|Ra|Rb|Si|si|se|Se|Sr|Te|te|Xe|Zn|>>|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\."
    r"|=|#|-|\+|\\\\|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
_MOLBOTTLE_RX = re.compile(_MOLBOTTLE_SMILES_PATTERN)


class MolBottleGenerator(FeatureGenerator):
    """Feature generator backed by a MolBottle ONNX encoder export.

    Reads ``config.json`` / ``vocab.json`` / ``encoder.onnx`` from ``export_dir``,
    defaulting to ``encoders/e_base`` next to this file. ``variant`` labels the
    generated feature names and defaults to the export directory's name, so
    different exports never collide even at the same training epoch.

    The encoder produces 250-dimensional float32 embeddings.  SMILES that
    cannot be tokenized or exceed ``max_len=220`` tokens are zero-filled and
    a warning is raised with the count — rows stay aligned with the input list.

    Requires: ``onnxruntime`` (``pip install onnxruntime``).
    """

    def __init__(
        self,
        export_dir: Optional[str] = None,
        batch_size: int = 100,
        variant: Optional[str] = None,
    ):
        super().__init__()

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for MolBottleGenerator. "
                "Install with: pip install onnxruntime"
            )

        base_dir = os.path.dirname(os.path.realpath(__file__))
        d = Path(export_dir) if export_dir is not None else Path(base_dir) / "encoders" / "e_base"

        cfg_path   = d / "config.json"
        vocab_path = d / "vocab.json"
        onnx_path  = d / "encoder.onnx"

        for p in (cfg_path, vocab_path, onnx_path):
            if not p.exists():
                raise FileNotFoundError(f"MolBottleGenerator: {p} not found")

        cfg = json.loads(cfg_path.read_text())
        tok_list: list = json.loads(vocab_path.read_text())["tok_list"]

        self._tok2int  = {t: i for i, t in enumerate(tok_list)}
        self._pad      = cfg["pad_index"]
        self._unk      = self._tok2int["<unk>"]
        self._sos      = self._tok2int["<sos>"]
        self._eos      = self._tok2int["<eos>"]
        self._max_len  = cfg["max_len"]
        self._onnx_path = str(onnx_path)
        self.batch_size = batch_size

        self._sess = ort.InferenceSession(self._onnx_path, providers=["CPUExecutionProvider"])
        self._out_name = self._sess.get_outputs()[0].name

        self.nb_features = cfg["out_dim"]
        epoch = cfg.get("source_epoch", "?")
        self.variant = variant if variant is not None else d.name
        self.names = [f"MolBottle_{i}_of_{self.nb_features}_{self.variant}_ep{epoch}"
                      for i in range(self.nb_features)]
        self.generator_name = f"automol_molbottle_{self.variant}_ep{epoch}"

    def _encode_one(self, smile: str) -> Optional[list]:
        """Tokenise one SMILES → padded int list, or None if unencodable."""
        if not isinstance(smile, str) or not smile:
            return None
        toks = _MOLBOTTLE_RX.findall(smile)
        if "".join(toks) != smile:
            return None
        seq = ([self._sos]
               + [self._tok2int.get(t, self._unk) for t in toks]
               + [self._eos])
        if len(seq) > self._max_len:
            return None
        return seq + [self._pad] * (self._max_len - len(seq))

    def generate(self, smiles) -> np.ndarray:
        """Return ``(n, 250)`` float32 embeddings aligned with the input list."""
        import warnings
        if hasattr(smiles, "tolist"):
            smiles = smiles.tolist()
        if isinstance(smiles, str):
            smiles = [smiles]

        X = np.zeros((len(smiles), self.nb_features), dtype=np.float32)
        rows = [(i, seq) for i, s in enumerate(smiles)
                if (seq := self._encode_one(s)) is not None]
        if len(rows) < len(smiles):
            warnings.warn(
                f"MolBottleGenerator: {len(smiles) - len(rows)} of {len(smiles)} "
                f"SMILES unparseable or exceeded max_len={self._max_len}; zero-filled"
            )
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start:start + self.batch_size]
            ids = np.array([seq for _, seq in chunk], dtype=np.int64)
            z = self._sess.run(
                [self._out_name],
                {"src": ids, "src_pad_mask": ids == self._pad},
            )[0]
            X[[i for i, _ in chunk]] = z
        X[~np.isfinite(X)] = 0.0
        return X

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sess"] = None
        state["_out_name"] = None
        state["_onnx_rel"] = _package_relative(self._onnx_path)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        import onnxruntime as ort
        self._onnx_path = _resolve_package_path(state.get("_onnx_rel"), state["_onnx_path"])
        self._sess = ort.InferenceSession(self._onnx_path, providers=["CPUExecutionProvider"])
        self._out_name = self._sess.get_outputs()[0].name

    def __repr__(self) -> str:
        return f"MolBottleGenerator(nb_features={self.nb_features}, generator='{self.generator_name}')"
