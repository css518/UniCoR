import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from prettytable import PrettyTable
from torch.nn.modules.activation import Tanh
from torch.autograd import Function
import copy
import logging

logger = logging.getLogger(__name__)

def whitening_torch_final(embeddings: torch.Tensor) -> torch.Tensor:
    """Apply whitening transformation to the input embeddings.
    Whitening normalizes the embeddings by removing correlation and scaling to unit variance.
    
    Args:
        embeddings: Input tensor of shape [batch_size, embedding_dim]
        
    Returns:
        torch.Tensor: Whitened embeddings of the same shape as input
    """
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings

def sigmas(model_name: str) -> List[float]:
    """Get pre-defined sigma values for different model architectures.
    
    These sigma values are used for the RBF kernel in MMD loss calculations.
    
    Args:
        model_name: Name of the model architecture
        
    Returns:
        List[float]: List of sigma values specific to the model
    """
    if 'unixcoder' in model_name.lower():
        return [0.5966, 1.1932, 2.3864]
    elif 'cocosoda' in model_name.lower():
        return [0.3835, 0.7669, 1.5338]
    elif 'graphcodebert' in model_name.lower():
        return [0.2853, 0.5705, 1.1411]
    elif 'codebert' in model_name.lower():
        return [0.1161, 0.2322, 0.4644]
    elif 'roberta' in model_name.lower():
        return [0.0417, 0.0833, 0.1667]
    else:
        logger.info('Bad Model Name')

class BaseModel(nn.Module): 
    """Base class for all models providing common functionality.
    
    This class extends nn.Module and provides methods for parameter inspection.
    """
    def __init__(self, ):
        super().__init__()
        
    def model_parameters(self) -> PrettyTable:
        """Generate a formatted table of model parameters.
        
        Returns:
            PrettyTable: A table containing layer names, output shapes, and parameter counts
        """
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

class Model_Eval(BaseModel):   
    """Model class for evaluation tasks.
    
    This class extends BaseModel and provides functionality for evaluating
    code representation models on various tasks.
    """
    def __init__(self, args: Any):
        """Initialize the evaluation model.
        
        Args:
            args: Configuration arguments containing model settings
        """
        super(Model_Eval, self).__init__()

        self.cls = args.cls
        self.model_name = args.model.lower()

        if self.model_name == 'zc3':
            self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
            config = RobertaConfig.from_pretrained('microsoft/codebert-base')
            model = RobertaModel.from_pretrained('microsoft/codebert-base', config=config) 
            self.encoder = Model_zc3(model, config, self.tokenizer)
            self.encoder.load_state_dict(torch.load('./chechpoints/ZC3/pytorch_model.bin', map_location=args.device), strict=False)
        elif self.model_name == 'bge':
            self.encoder = AutoModel.from_pretrained(args.model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
            self.encoder = RobertaModel.from_pretrained(args.model_name_or_path)

        if 'CodeBert' in args.model or 'ZC3' in args.model:
            self.cls = True
      
    def forward(self, inputs: Optional[torch.Tensor] = None, 
                 attn_mask: Optional[torch.Tensor] = None, 
                 position_idx: Optional[torch.Tensor] = None) -> torch.Tensor: 
        """Forward pass to compute embeddings for input sequences.
        
        Depending on configuration, computes either [CLS] token embeddings 
        or average pooling over token embeddings.
        
        Args:
            inputs: Input token ids tensor of shape [batch_size, seq_length]
            attn_mask: Attention mask tensor of shape [batch_size, seq_length]
            position_idx: Position indices tensor for graph structures
            
        Returns:
            torch.Tensor: Normalized embeddings of shape [batch_size, embedding_dim]
        """
        if self.cls:
            if attn_mask is None:
                attn_mask = inputs.ne(1)
                if self.model_name == 'zc3':
                    outputs = self.encoder.encoder(inputs,attention_mask=attn_mask)[1]
                else:
                    outputs = self.encoder(inputs,attention_mask=attn_mask)[1]
                    outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            else:
                nodes_mask=position_idx.eq(0)
                token_mask=position_idx.ge(2)        
                inputs=self.encoder.embeddings.word_embeddings(inputs)
                nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
                nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
                avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs)
                inputs=inputs*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
                outputs = self.encoder(inputs_embeds=inputs,attention_mask=attn_mask,position_ids=position_idx)[1]
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            return outputs
        else:
            if self.model_name == 'bge':
                outputs = self.encoder(inputs)[0][:, 0]
            else:
                outputs = self.encoder(inputs,attention_mask=inputs.ne(1))[0]
                outputs = (outputs*inputs.ne(1)[:,:,None]).sum(1)/inputs.ne(1).sum(-1)[:,None] 
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
    
class MMDLoss(nn.Module):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between two sets of features.
    Uses an RBF kernel by default. Can be configured for multiple kernels.
    Implements the unbiased estimator for MMD^2.

    Args:
        kernel_type (str): Type of kernel. 'rbf' (default) or 'poly'.
        kernel_mul (float): Factor to create multiple sigmas for RBF kernel.
                            Effective if `sigmas` and `fix_sigma` are None.
                            sigmas are `base_sigma * (kernel_mul^i)`.
        kernel_num (int): Number of RBF kernels to use (each with a different sigma).
                          Effective if `sigmas` and `fix_sigma` are None.
        sigmas (list of float, optional): Explicitly define a list of sigmas for RBF kernels.
                                         Overrides `kernel_mul` and `kernel_num`.
        fix_sigma (float, optional): Use a single fixed sigma for the RBF kernel.
                                     Overrides `kernel_mul`, `kernel_num`, and `sigmas`.
        base_sigma_heuristic (str, optional): Heuristic to determine a base sigma if not fixed.
                                          Currently, only 'median' or None is indicative.
                                          If 'median', median pairwise distance is used (can be slow).
                                          If None, a default base (e.g., 1.0 or feature_dim) might be used.
                                          For simplicity, this implementation will use a simpler default if not specified.
        **kwargs: Additional arguments for specific kernels (e.g., `degree`, `coef0` for 'poly').
    """
    def __init__(self, kernel_type: str = 'rbf', kernel_mul: float = 2.0, kernel_num: int = 5,
                 sigmas: Optional[List[float]] = None, fix_sigma: Optional[float] = None, 
                 base_sigma_heuristic: Optional[str] = None, **kwargs: Dict[str, Any]):
        """Initialize the MMD loss module.
        
        Args:
            kernel_type: Type of kernel. 'rbf' (default) or 'poly'.
            kernel_mul: Factor to create multiple sigmas for RBF kernel.
            kernel_num: Number of RBF kernels to use (each with a different sigma).
            sigmas: Explicitly define a list of sigmas for RBF kernels.
            fix_sigma: Use a single fixed sigma for the RBF kernel.
            base_sigma_heuristic: Heuristic to determine a base sigma if not fixed.
            **kwargs: Additional arguments for specific kernels.
        """
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type.lower()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self._sigmas_list = [] # To be populated based on priority

        if sigmas is not None:
            if isinstance(sigmas, list) and len(sigmas) > 0:
                self._sigmas_list = sigmas
            else:
                self._sigmas_list = [fix_sigma]
            
        # Else, sigmas will be determined dynamically in the forward pass if RBF kernel is used
        # and _sigmas_list is empty, or a default will be used.

        self.base_sigma_heuristic = base_sigma_heuristic # For potential future use of median heuristic
        self.kernel_args = kwargs

    def _rbf_kernel_gram_matrix(self, X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Computes the RBF Gram matrix between X and Y.
        K_ij = exp(- ||X_i - Y_j||^2 / (2 * sigma^2))
        
        Args:
            X: First set of feature vectors of shape [m, d]
            Y: Second set of feature vectors of shape [n, d]
            sigma: Kernel width parameter
            
        Returns:
            torch.Tensor: Gram matrix of shape [m, n]
        """
        # torch.cdist computes pairwise p-norm distances. p=2 for Euclidean.
        # Resulting shape: (X.size(0), Y.size(0))
        dists_sq = torch.cdist(X, Y, p=2).pow(2)
        return torch.exp(-dists_sq / (2 * sigma**2))

    def _polynomial_kernel_gram_matrix(self, X: torch.Tensor, Y: torch.Tensor, 
                                       degree: int, gamma: Optional[float], 
                                       coef0: float) -> torch.Tensor:
        """
        Computes the Polynomial Gram matrix.
        K_ij = (gamma * <X_i, Y_j> + coef0)^degree
        
        Args:
            X: First set of feature vectors of shape [m, d]
            Y: Second set of feature vectors of shape [n, d]
            degree: Degree of the polynomial kernel
            gamma: Kernel coefficient (if None, 1/d is used)
            coef0: Independent term in the kernel
            
        Returns:
            torch.Tensor: Gram matrix of shape [m, n]
        """
        if gamma is None: # Default gamma if not provided
            gamma = 1.0 / X.size(1) # 1/feature_dim
        return (gamma * (X @ Y.t()) + coef0).pow(degree)

    def _calculate_mmd2_unbiased(self, K_XX: torch.Tensor, K_YY: torch.Tensor, 
                                K_XY: torch.Tensor, m: int, n: int) -> torch.Tensor:
        """
        Calculates MMD^2 using the unbiased estimator.
        
        Args:
            K_XX: Kernel matrix for samples from distribution X
            K_YY: Kernel matrix for samples from distribution Y
            K_XY: Cross kernel matrix between X and Y samples
            m: Number of samples from distribution X
            n: Number of samples from distribution Y
            
        Returns:
            torch.Tensor: Unbiased MMD^2 estimate
        """
        # Term 1: E[k(X, X')]
        if m > 1:
            # Sum off-diagonal elements for K_XX
            term1 = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1))
        else:
            # If m <= 1, this term is conventionally 0 for the unbiased estimator
            term1 = torch.tensor(0.0, device=K_XX.device, dtype=K_XX.dtype)

        # Term 2: E[k(Y, Y')]
        if n > 1:
            # Sum off-diagonal elements for K_YY
            term2 = (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1))
        else:
            # If n <= 1, this term is conventionally 0
            term2 = torch.tensor(0.0, device=K_YY.device, dtype=K_YY.dtype)

        # Term 3: -2 * E[k(X, Y)]
        # .mean() is equivalent to .sum() / (m * n)
        if m > 0 and n > 0:
            term3 = -2 * K_XY.mean()
        else:
            # If either m or n is 0, this term is 0
            term3 = torch.tensor(0.0, device=K_XY.device, dtype=K_XY.dtype)

        mmd2 = term1 + term2 + term3
        return mmd2

    def _get_rbf_sigmas(self, X: torch.Tensor, Y: torch.Tensor) -> List[float]:
        """Determine sigmas for RBF kernel if not pre-defined.
        
        Args:
            X: First set of feature vectors
            Y: Second set of feature vectors
            
        Returns:
            List[float]: List of sigma values to use in RBF kernels
        """
        if self._sigmas_list: # Already defined by user (fix_sigma or sigmas list)
            return self._sigmas_list

        # Fallback: if no sigmas are provided via constructor, create a default list
        # A common heuristic is using the median pairwise distance, but this can be slow
        # and requires care if X or Y is empty or has one element.
        # For simplicity in this example, if not provided, use a default sigma range.
        # A more robust solution involves careful median heuristic or making sigma a learned parameter.
        if self.base_sigma_heuristic == 'median':
            # Caution: Median heuristic can be slow and needs careful implementation for edge cases
            # For example:
            # Z = torch.cat([X,Y], dim=0)
            # if Z.size(0) < 2: return [1.0]
            # dists = torch.pdist(Z, p=2)
            # median_dist = torch.median(dists)
            # base_sigma = median_dist.item() if median_dist > 1e-7 else 1.0
            # This is a placeholder for a more robust median heuristic
            print("Warning: Median heuristic for sigma is indicative and might be slow or default.")
            base_sigma = 1.0 # Fallback if median calculation is not robustly implemented here
        else:
            # Fallback to a simpler method: use feature dimension or a fixed value
            # base_sigma = X.size(1) # e.g., feature_dim
            base_sigma = 1.0 # A common simple default

        if self.kernel_num == 1:
            return [base_sigma]

        sigmas = [base_sigma * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        # Ensure sigmas are positive and distinct enough
        sigmas = [s for s in sigmas if s > 1e-7]
        if not sigmas: sigmas = [1.0] # Fallback
        return list(set(sigmas)) # Ensure unique sigmas


    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor):
        """
        Args:
            source_features: torch.Tensor of shape (batch_size_source, feature_dim)
                             Embeddings from UniXcoder for the source language.
            target_features: torch.Tensor of shape (batch_size_target, feature_dim)
                             Embeddings from UniXcoder for the target language.
        Returns:
            mmd_loss: scalar MMD^2 loss value
        """
        # Handle empty inputs: if either set of features is empty, MMD is typically 0 or undefined.
        # Returning 0 is a common practical choice.
        if source_features.size(0) == 0 or target_features.size(0) == 0:
            return torch.tensor(0.0, device=source_features.device if source_features.size(0) > 0 else target_features.device,
                                dtype=source_features.dtype if source_features.size(0) > 0 else target_features.dtype)

        m, n = source_features.size(0), target_features.size(0)

        if self.kernel_type == 'rbf':
            # Determine sigmas to use
            sigmas_to_use = self._get_rbf_sigmas(source_features, target_features)
            if not sigmas_to_use: # Should not happen if _get_rbf_sigmas has fallback
                return torch.tensor(0.0, device=source_features.device, dtype=source_features.dtype)

            total_mmd2 = torch.tensor(0.0, device=source_features.device, dtype=source_features.dtype)
            for sigma_val in sigmas_to_use:
                if sigma_val <= 1e-7: continue # Sigma must be positive

                K_XX = self._rbf_kernel_gram_matrix(source_features, source_features, sigma_val)
                K_YY = self._rbf_kernel_gram_matrix(target_features, target_features, sigma_val)
                K_XY = self._rbf_kernel_gram_matrix(source_features, target_features, sigma_val)
                total_mmd2 += self._calculate_mmd2_unbiased(K_XX, K_YY, K_XY, m, n)

            mmd2_final = total_mmd2 / len(sigmas_to_use)

        elif self.kernel_type == 'poly':
            degree = self.kernel_args.get('degree', 3)
            gamma_poly = self.kernel_args.get('gamma_poly', None) # If None, will be 1/feature_dim in _polynomial_kernel_gram_matrix
            coef0 = self.kernel_args.get('coef0', 1.0)

            K_XX = self._polynomial_kernel_gram_matrix(source_features, source_features, degree, gamma_poly, coef0)
            K_YY = self._polynomial_kernel_gram_matrix(target_features, target_features, degree, gamma_poly, coef0)
            K_XY = self._polynomial_kernel_gram_matrix(source_features, target_features, degree, gamma_poly, coef0)
            mmd2_final = self._calculate_mmd2_unbiased(K_XX, K_YY, K_XY, m, n)
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}. Supported: 'rbf', 'poly'.")

        # MMD^2 should be non-negative. Clamp at 0 due to potential floating point inaccuracies leading to small negative values.
        return torch.relu(mmd2_final)
        
class UniCoR(BaseModel):
    """UniCoR (Unified Contrastive Representation) model for code representation learning.
    
    This model extends BaseModel and implements contrastive learning between code and natural language,
    with support for multiple loss functions and data augmentation techniques.
    """
    def __init__(self, base_encoder: nn.Module, args: Any):
        """Initialize the UniCoR model.
        
        Args:
            base_encoder: Base transformer encoder model
            args: Configuration arguments containing model settings
        """
        super(UniCoR, self).__init__()

        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        dim= args.moco_dim
        self.mlp = args.mlp

        self.cls = args.cls
        self.code_encoder_q = base_encoder
        self.code_encoder_k = copy.deepcopy(base_encoder)
        self.nl_encoder_q = base_encoder
        self.nl_encoder_k = copy.deepcopy(self.nl_encoder_q)

        self.mmdloss_fn = MMDLoss(kernel_type='rbf', sigmas=sigmas(args.model.lower()))
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.cl_loss_fn = nn.CrossEntropyLoss()

        self.time_score= args.time_score
        self.do_whitening = args.do_whitening
        self.do_inter_loss = args.do_inter_loss
        self.do_aug_loss = args.do_aug_loss
        self.do_inner_code_loss = args.do_inner_code_loss 
        self.do_inner_nl_loss = args.do_inner_nl_loss 
        self.do_mmd_loss = args.do_mmd_loss
        self.do_kl_loss = args.do_kl_loss
        self.args = args

        for param_q, param_k in zip(self.code_encoder_q.parameters(), self.code_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.nl_encoder_q.parameters(), self.nl_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the code queue
        torch.manual_seed(3047)
        torch.cuda.manual_seed(3047)
        self.register_buffer("code_queue", torch.randn(dim,self.K))
        self.code_queue = nn.functional.normalize(self.code_queue, dim=0)
        self.register_buffer("code_queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the masked code queue
        self.register_buffer("masked_code_queue", torch.randn(dim, self.K))
        self.masked_code_queue = nn.functional.normalize(self.masked_code_queue, dim=0)
        self.register_buffer("masked_code_queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the nl queue
        self.register_buffer("nl_queue", torch.randn(dim, self.K))
        self.nl_queue = nn.functional.normalize(self.nl_queue, dim=0)
        self.register_buffer("nl_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("masked_nl_queue", torch.randn(dim, self.K))
        self.masked_nl_queue= nn.functional.normalize(self.masked_nl_queue, dim=0)
        self.register_buffer("masked_nl_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Update the key encoder parameters using momentum.
        
        This method performs a momentum update of the key encoder parameters
        based on the query encoder parameters, which helps stabilize training.
        """
        for param_q, param_k in zip(self.code_encoder_q.parameters(), self.code_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.nl_encoder_q.parameters(), self.nl_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if self.mlp:
            for param_q, param_k in zip(self.code_encoder_q_fc.parameters(), self.code_encoder_k_fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            for param_q, param_k in zip(self.nl_encoder_q_fc.parameters(), self.nl_encoder_k_fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, option: str = 'code') -> None:
        """Update the feature queues by removing old entries and adding new ones.
        
        Args:
            keys: Batch of feature vectors to enqueue
            option: Type of queue to update ('code', 'masked_code', 'nl', or 'masked_nl')
        """
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        if option == 'code':
            code_ptr = int(self.code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.code_queue[:, code_ptr:code_ptr + batch_size] = keys.T
            except:
                print(code_ptr)
                print(batch_size)
                print(keys.shape)
                exit(111)
            code_ptr = (code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.code_queue_ptr[0] = code_ptr
        
        elif option == 'masked_code':
            masked_code_ptr = int(self.masked_code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.masked_code_queue[:, masked_code_ptr:masked_code_ptr + batch_size] = keys.T
            except:
                print(masked_code_ptr)
                print(batch_size)
                print(keys.shape)
                exit(111)
            masked_code_ptr = (masked_code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.masked_code_queue_ptr[0] = masked_code_ptr
        
        elif option == 'nl':

            nl_ptr = int(self.nl_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.nl_queue[:, nl_ptr:nl_ptr + batch_size] = keys.T
            nl_ptr = (nl_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.nl_queue_ptr[0] = nl_ptr
        
        elif option == 'masked_nl':
    
            masked_nl_ptr = int(self.masked_nl_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.masked_nl_queue[:, masked_nl_ptr:masked_nl_ptr + batch_size] = keys.T
            masked_nl_ptr = (masked_nl_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.masked_nl_queue_ptr[0] = masked_nl_ptr

    def do_repre(self, encoder: nn.Module, data: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None, 
                 pe: Optional[torch.Tensor] = None, 
                 iscode: bool = False) -> torch.Tensor:
        """Compute normalized representations from the encoder.
        
        Depending on the input type and configuration, this method computes
        feature embeddings and applies normalization or whitening.
        
        Args:
            encoder: Model encoder to use for representation extraction
            data: Input tensor of token ids or embeddings
            mask: Attention mask tensor
            pe: Position encoding tensor
            iscode: Whether the input is code (affects processing)
            
        Returns:
            torch.Tensor: Normalized feature embeddings
        """
        if iscode and mask is not None:
            nodes_mask=pe.eq(0)
            token_mask=pe.ge(2)        
            inputs_embeddings=encoder.embeddings.word_embeddings(data)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            outputs = encoder(inputs_embeds=inputs_embeddings,attention_mask=mask,position_ids=pe)[1]
        else:
            outputs = encoder(data,attention_mask=data.ne(1))[1]

        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        if self.do_whitening:
            outputs = whitening_torch_final(outputs)
        return outputs

    def do_contrasive_loss(self, query: torch.Tensor, pos: torch.Tensor, 
                           negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute contrastive loss for the given query and key representations.
        
        This method calculates the contrastive loss by computing similarities between
        query vectors, positive keys, and negative keys.
        
        Args:
            query: Query feature vectors of shape [batch_size, feature_dim]
            pos: Positive key vectors of shape [batch_size, feature_dim]
            negative: Negative key vectors of shape [feature_dim, queue_size]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits tensor and corresponding labels tensor
        """
        logits_pos = torch.einsum('nc,bc->nb', [query, pos])
        # negative logits: NxK
        logits_neg = torch.einsum('nc,ck->nk', [query, negative])

        logits = torch.cat([self.time_score*logits_pos, logits_neg], dim=1)
        logits /= self.T
        label = torch.arange(logits.size(0), device=logits.device)
        return logits, label

    def forward(self, code1_q_r: Optional[torch.Tensor] = None, 
                code1_k_r: Optional[torch.Tensor] = None, 
                code2_q_r: Optional[torch.Tensor] = None, 
                code2_k_r: Optional[torch.Tensor] = None, 
                nl1_q_r: Optional[torch.Tensor] = None, 
                nl1_k_r: Optional[torch.Tensor] = None, 
                nl2_q_r: Optional[torch.Tensor] = None, 
                nl2_k_r: Optional[torch.Tensor] = None,
                code1_mask: Optional[torch.Tensor] = None, 
                code1_pe: Optional[torch.Tensor] = None, 
                code2_mask: Optional[torch.Tensor] = None, 
                code2_pe: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward of the UniCoR model.
        
        This method computes various loss terms for contrastive learning between
        code and natural language representations.
        
        Args:
            code1_q_r: Code 1 query token ids
            code1_k_r: Code 1 key token ids
            code2_q_r: Code 2 query token ids
            code2_k_r: Code 2 key token ids
            nl1_q_r: Natural language 1 query token ids
            nl1_k_r: Natural language 1 key token ids
            nl2_q_r: Natural language 2 query token ids
            nl2_k_r: Natural language 2 key token ids
            code1_mask: Code 1 attention mask
            code1_pe: Code 1 position encoding
            code2_mask: Code 2 attention mask
            code2_pe: Code 2 position encoding
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: 
                Contrastive loss, MMD loss, and KL divergence loss
        """
        code1_q = self.do_repre(self.code_encoder_q, code1_q_r, code1_mask, code1_pe, iscode=True) # compute query features for source code1
        code2_q = self.do_repre(self.code_encoder_q, code2_q_r, code2_mask, code2_pe, iscode=True)
        nl1_q = self.do_repre(self.nl_encoder_q, nl1_q_r) # compute query features for nl
        nl2_q = self.do_repre(self.nl_encoder_q, nl2_q_r)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            code1_k = self.do_repre(self.code_encoder_k, code1_k_r, code1_mask, code1_pe, iscode=True) # compute keys features for source code1
            code2_k = self.do_repre(self.code_encoder_k, code2_k_r, code2_mask, code2_pe, iscode=True) # compute keys features for source code2
            nl1_k = self.do_repre(self.nl_encoder_k, nl1_k_r) # compute keys features for nl
            nl2_k = self.do_repre(self.nl_encoder_k, nl2_k_r)


        if self.do_mmd_loss:
            mmd_loss_local = self.mmdloss_fn(code1_q, code2_q)
            mmd_loss_global1 = self.mmdloss_fn(code1_q, self.code_queue.clone().detach().t())
            mmd_loss_global2 = self.mmdloss_fn(code2_q, self.code_queue.clone().detach().t())
            mmd_loss = mmd_loss_local + mmd_loss_global1 + mmd_loss_global2
        else:
            mmd_loss = None
        
        if self.do_kl_loss:
            with torch.no_grad():
                code1_q_old = self.do_repre(self.code_encoder_k, code1_q_r) # compute keys features for source code1
                code2_q_old = self.do_repre(self.code_encoder_k, code2_q_r)
            kl_loss = self.kl_loss_fn(F.log_softmax(code1_q), F.softmax(code1_q_old)) + self.kl_loss_fn(F.log_softmax(code2_q), F.softmax(code2_q_old))
        else:
            kl_loss = None

        #Normal Loss: Raw pair -> Code vs NL
        code2nl_logits11, code2nl_label11 = self.do_contrasive_loss(code1_q, nl1_q, self.nl_queue.clone().detach())# code1 vs nl1
        code2nl_logits22, code2nl_label22 = self.do_contrasive_loss(code2_q, nl2_q, self.nl_queue.clone().detach())## code2 vs nl2
        
        nl2code_logits11, nl2code_label11 = self.do_contrasive_loss(nl1_q, code1_q, self.code_queue.clone().detach())## nl1 vs code1
        nl2code_logits22, nl2code_label22 = self.do_contrasive_loss(nl2_q, code2_q, self.code_queue.clone().detach())## nl2 vs code2

        if self.do_aug_loss:
            code2maskednl_logits11, code2maskednl_label11 = self.do_contrasive_loss(code1_q, nl1_k, self.masked_nl_queue.clone().detach())## code1 vs masked nl1
            code2maskednl_logits22, code2maskednl_label22 = self.do_contrasive_loss(code2_q, nl2_k, self.masked_nl_queue.clone().detach())## code2 vs masked nl2
            nl2maskedcode_logits11, nl2maskedcode_label11 = self.do_contrasive_loss(nl1_q, code1_k, self.masked_code_queue.clone().detach())
            nl2maskedcode_logits22, nl2maskedcode_label22 = self.do_contrasive_loss(nl2_q, code2_k, self.masked_code_queue.clone().detach()) ## nl2 vs masked code2

            inter_code2nl_logits = torch.cat((code2nl_logits11, code2maskednl_logits11, code2nl_logits22 ,code2maskednl_logits22), dim=0)
            inter_code2nl_labels = torch.cat((code2nl_label11, code2maskednl_label11, code2nl_label22 ,code2maskednl_label22), dim=0)
            inter_nl2code_logits = torch.cat((nl2code_logits11, nl2maskedcode_logits11 ,nl2code_logits22, nl2maskedcode_logits22), dim=0)
            inter_nl2code_labels = torch.cat((nl2code_label11, nl2maskedcode_label11 ,nl2code_label22, nl2maskedcode_label22), dim=0)

        else:
            inter_code2nl_logits = torch.cat((code2nl_logits11, code2nl_logits22), dim=0)
            inter_code2nl_labels = torch.cat((code2nl_label11, code2nl_label22), dim=0)
            inter_nl2code_logits = torch.cat((nl2code_logits11 ,nl2code_logits22), dim=0)
            inter_nl2code_labels = torch.cat((nl2code_label11 ,nl2code_label22), dim=0)

        #Inter Loss: different language pair -> Code vs NL
        if self.do_inter_loss:
            code2nl_logits12, code2nl_label12 = self.do_contrasive_loss(code1_q, nl2_q, self.nl_queue.clone().detach())  #code1 vs nl2
            code2nl_logits21, code2nl_label21 = self.do_contrasive_loss(code2_q, nl1_q, self.nl_queue.clone().detach())  #code2 vs nl1
            nl2code_logits12, nl2code_label12 = self.do_contrasive_loss(nl1_q, code2_q, self.code_queue.clone().detach()) #nl1 vs code2
            nl2code_logits21, nl2code_label21 = self.do_contrasive_loss(nl2_q, code1_q, self.code_queue.clone().detach()) #nl2 vs code1

            if self.do_aug_loss:
                code2maskednl_logits12, code2maskednl_label12 = self.do_contrasive_loss(code1_q, nl2_k, self.masked_nl_queue.clone().detach())## code1 vs masked nl2
                code2maskednl_logits21, code2maskednl_label21 = self.do_contrasive_loss(code2_q, nl1_k, self.masked_nl_queue.clone().detach())## code2 vs masked nl1
                nl2maskedcode_logits12, nl2maskedcode_label12 = self.do_contrasive_loss(nl1_q, code2_k, self.masked_code_queue.clone().detach()) ## nl1 vs masked code2
                nl2maskedcode_logits21, nl2maskedcode_label21 = self.do_contrasive_loss(nl2_q, code1_k, self.masked_code_queue.clone().detach()) ## nl2 vs masked code1

                inter_code2nl_logits = torch.cat((inter_code2nl_logits, code2nl_logits12, code2nl_logits21, code2maskednl_logits12, code2maskednl_logits21), dim=0)
                inter_code2nl_labels = torch.cat((inter_code2nl_labels, code2nl_label12, code2nl_label21, code2maskednl_label12, code2maskednl_label21), dim=0)
                inter_nl2code_logits = torch.cat((inter_nl2code_logits, nl2code_logits12, nl2code_logits21, nl2maskedcode_logits12, nl2maskedcode_logits21), dim=0)
                inter_nl2code_labels = torch.cat((inter_nl2code_labels, nl2code_label12, nl2code_label21, nl2maskedcode_label12, nl2maskedcode_label21), dim=0)   

            else:
                inter_code2nl_logits = torch.cat((inter_code2nl_logits, code2nl_logits12, code2nl_logits21), dim=0)
                inter_code2nl_labels = torch.cat((inter_code2nl_labels, code2nl_label12, code2nl_label21), dim=0)
                inter_nl2code_logits = torch.cat((inter_nl2code_logits, nl2code_logits12, nl2code_logits21), dim=0)
                inter_nl2code_labels = torch.cat((inter_nl2code_labels, nl2code_label12, nl2code_label21), dim=0)      
        
        inner_code_logits, inner_nl_logits = None, None
        #Inner Loss: -> Code vs Code
        if self.do_inner_code_loss:
            if self.do_aug_loss:
                code2maskedcode_logits11, code2maskedcode_label11 = self.do_contrasive_loss(code1_q, code1_k, self.masked_code_queue.clone().detach())
                code2maskedcode_logits22, code2maskedcode_label22 = self.do_contrasive_loss(code2_q, code2_k, self.masked_code_queue.clone().detach())

                inner_code_logits = torch.cat((code2maskedcode_logits11, code2maskedcode_logits22), dim=0)
                inner_code_labels = torch.cat((code2maskedcode_label11, code2maskedcode_label22), dim=0)

            if self.do_inter_loss:
                code2code_logits12, code2code_label12 = self.do_contrasive_loss(code1_q, code2_q, self.code_queue.clone().detach())
                code2code_logits21, code2code_label21 = self.do_contrasive_loss(code2_q, code1_q, self.code_queue.clone().detach())

                if inner_code_logits is not None:
                    code2maskedcode_logits21, code2maskedcode_label21 = self.do_contrasive_loss(code2_q, code1_k, self.masked_code_queue.clone().detach())
                    code2maskedcode_logits12, code2maskedcode_label12 = self.do_contrasive_loss(code1_q, code2_k, self.masked_code_queue.clone().detach())
                    
                    inner_code_logits = torch.cat((inner_code_logits, code2code_logits12, code2code_logits21, code2maskedcode_logits21, code2maskedcode_logits12), dim=0)
                    inner_code_labels = torch.cat((inner_code_labels, code2code_label12, code2code_label21, code2maskedcode_label21, code2maskedcode_label12), dim=0)
                else:
                    inner_code_logits = torch.cat((code2code_logits12, code2code_logits21), dim=0)
                    inner_code_labels = torch.cat((code2code_label12, code2code_label21), dim=0)
        
        #Inner Loss: -> NL vs NL
        if self.do_inner_nl_loss:
            if self.do_aug_loss:
                nl2maskednl_logits11, nl2maskednl_label11 = self.do_contrasive_loss(nl1_q, nl1_k, self.masked_nl_queue.clone().detach())
                nl2maskednl_logits22, nl2maskednl_label22 = self.do_contrasive_loss(nl2_q, nl2_k, self.masked_nl_queue.clone().detach())
            
                inner_nl_logits = torch.cat((nl2maskednl_logits11, nl2maskednl_logits22), dim=0)
                inner_nl_labels = torch.cat((nl2maskednl_label11, nl2maskednl_label22), dim=0)

            if self.do_inter_loss:
                nl2nl_logits12, nl2nl_label12 = self.do_contrasive_loss(nl1_q, nl2_q, self.nl_queue.clone().detach())
                nl2nl_logits21, nl2nl_label21 = self.do_contrasive_loss(nl2_q, nl1_q, self.nl_queue.clone().detach())

                if inner_nl_logits is not None:
                    nl2maskednl_logits12, nl2maskednl_label12 = self.do_contrasive_loss(nl1_q, nl2_k, self.masked_nl_queue.clone().detach())
                    nl2maskednl_logits21, nl2maskednl_label21 = self.do_contrasive_loss(nl2_q, nl1_k, self.masked_nl_queue.clone().detach())
                    inner_nl_logits = torch.cat((inner_nl_logits, nl2nl_logits12, nl2nl_logits21, nl2maskednl_logits12, nl2maskednl_logits21), dim=0)
                    inner_nl_labels = torch.cat((inner_nl_labels, nl2nl_label12, nl2nl_label21, nl2maskednl_label12, nl2maskednl_label21), dim=0)
                else:
                    inner_nl_logits = torch.cat((nl2nl_logits12, nl2nl_logits21), dim=0)
                    inner_nl_labels = torch.cat((nl2nl_label12, nl2nl_label21), dim=0)

        logits = torch.cat((inter_code2nl_logits, inter_nl2code_logits), dim=0)
        lables = torch.cat((inter_code2nl_labels, inter_nl2code_labels), dim=0)

        if inner_code_logits is not None:
            logits = torch.cat((logits, inner_code_logits), dim=0)
            lables = torch.cat((lables, inner_code_labels), dim=0)
        
        if inner_nl_logits is not None:
            logits = torch.cat((logits, inner_nl_logits), dim=0)
            lables = torch.cat((lables, inner_nl_labels), dim=0)

        loss_cl = self.cl_loss_fn(20*logits, lables)

        # dequeue and enqueue
        self._dequeue_and_enqueue(code1_q, option='code')
        self._dequeue_and_enqueue(code2_q, option='code')
        self._dequeue_and_enqueue(nl1_q, option='nl')
        self._dequeue_and_enqueue(nl2_q, option='nl')
        self._dequeue_and_enqueue(code1_k, option='masked_code')
        self._dequeue_and_enqueue(code2_k, option='masked_code')
        self._dequeue_and_enqueue(nl1_k, option='masked_nl')
        self._dequeue_and_enqueue(nl2_k, option='masked_nl')

        return loss_cl, mmd_loss, kl_loss
            