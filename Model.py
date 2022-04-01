import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import pykp
from pykp.modules import RNNEncoder, RNNDecoder


class nTSNTM(nn.Module):

    def __init__(self,
                 prior_alpha,
                 prior_beta,
                 vocab_size,
                 n_hidden,
                 truncation_level,
                 n_topic2,
                 learning_rate,
                 batch_size,
                 non_linearity,
                 device):
        super(nTSNTM, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = truncation_level
        self.n_topic2 = n_topic2
        self.n_topic3 = 1
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if non_linearity == 'tanh':
            non_linearity = nn.Tanh()
        elif non_linearity == 'sigmoid':
            non_linearity = nn.Sigmoid()
        else:
            non_linearity = nn.ReLU()
        "encoder parameters"
        self.mlp = nn.Sequential(
            nn.Linear(vocab_size, n_hidden),
            non_linearity
        )
        self.a = nn.Linear(self.n_hidden, self.n_topic)
        self.b = nn.Linear(self.n_hidden, self.n_topic)
        self.mean = nn.Linear(self.n_hidden, self.n_hidden)
        self.logsigm = nn.Linear(self.n_hidden, self.n_hidden)
        "decoder parameters"
        self.eta = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            non_linearity,
            nn.Linear(n_hidden, 3),
            non_linearity
        )
        self.remaining_stick = torch.ones(self.batch_size)
        self.params = nn.ParameterDict({
            'word_vec': nn.Parameter(torch.Tensor(self.vocab_size, self.n_hidden)),
            'topic_vec': nn.Parameter(torch.Tensor(self.n_topic, self.n_hidden)),
            'depend': nn.Parameter(torch.Tensor(self.n_topic, self.n_topic2)),
            'topic_vec2': nn.Parameter(torch.Tensor(self.n_topic2, self.n_hidden)),
            'depend2': nn.Parameter(torch.Tensor(self.n_topic2, self.n_topic3)),
            'topic_vec3': nn.Parameter(torch.Tensor(self.n_topic3, self.n_hidden))
        })
        self.weight_init()

        # self.word_vec = nn.Parameter(torch.Tensor(self.vocab_size, self.n_hidden).uniform_(-limit1, limit1))
        # self.topic_vec = nn.Parameter(torch.Tensor(self.n_topic, self.n_hidden).uniform_(-limit2, limit2))
        # self.depend = nn.Parameter(torch.Tensor(self.n_topic, self.n_topic2).uniform_(-limit3, limit3))
        # self.topic_vec2 = nn.Parameter(torch.Tensor(self.n_topic2, self.n_hidden).uniform_(-limit4, limit4))
        # self.depend2 = nn.Parameter(torch.Tensor(self.n_topic2, self.n_topic3).uniform_(-limit5, limit5))
        # self.topic_vec3 = nn.Parameter(torch.Tensor(self.n_topic3, self.n_hidden).uniform_(-limit6, limit6))

    def weight_init(self):
        limit1 = math.sqrt(6 / (self.vocab_size + self.n_hidden))
        limit2 = math.sqrt(6 / (self.n_topic + self.n_hidden))
        limit3 = math.sqrt(6 / (self.n_topic + self.n_topic2))
        limit4 = math.sqrt(6 / (self.n_topic2 + self.n_hidden))
        limit5 = math.sqrt(6 / (self.n_topic2 + self.n_topic3))
        limit6 = math.sqrt(6 / (self.n_topic3 + self.n_hidden))
        nn.init.uniform_(self.params['word_vec'], -limit1, limit1)
        nn.init.uniform_(self.params['topic_vec'], -limit2, limit2)
        nn.init.uniform_(self.params['depend'], -limit3, limit3)
        nn.init.uniform_(self.params['topic_vec2'], -limit4, limit4)
        nn.init.uniform_(self.params['depend2'], -limit5, limit5)
        nn.init.uniform_(self.params['topic_vec3'], -limit6, limit6)

    def encode(self, x):
        enc_vec = self.mlp(x)
        a = F.softplus(self.a(enc_vec))
        b = F.softplus(self.b(enc_vec))
        mean = self.mean(enc_vec)
        logsigm = self.logsigm(enc_vec)
        kld_gauss = -0.5 * torch.sum(1 - torch.square(mean) + 2 * logsigm - torch.exp(2 * logsigm), 1)
        kl = 1. / (1 + a * b) * Beta_fn(1. / a, b)
        kl += 1. / (2 + a * b) * Beta_fn(2. / a, b)
        kl += 1. / (3 + a * b) * Beta_fn(3. / a, b)
        kl += 1. / (4 + a * b) * Beta_fn(4. / a, b)
        kl += 1. / (5 + a * b) * Beta_fn(5. / a, b)
        kl += 1. / (6 + a * b) * Beta_fn(6. / a, b)
        kl += 1. / (7 + a * b) * Beta_fn(7. / a, b)
        kl += 1. / (8 + a * b) * Beta_fn(8. / a, b)
        kl += 1. / (9 + a * b) * Beta_fn(9. / a, b)
        kl += 1. / (10 + a * b) * Beta_fn(10. / a, b)
        kl *= (self.prior_beta - 1) * b
        psi_b_taylor_approx = torch.digamma(b)
        kl += (a - self.prior_alpha) / a * (
                -0.57721 - psi_b_taylor_approx - 1 / b)  # T.psi(self.posterior_b)
        kl += torch.log(a * b) + torch.log(torch.exp(torch.lgamma(torch.Tensor([self.prior_alpha])).to(self.device) +
                                                     torch.lgamma(torch.Tensor([self.prior_beta])).to(self.device) -
                                                     torch.lgamma(torch.Tensor([self.prior_alpha + self.prior_beta]).to(
                                                         self.device))))
        kl += -(b - 1) / b
        kld_ku = torch.sum(kl, 1)
        kld = (kld_ku + kld_gauss)
        return mean, logsigm, kld, a, b

    def decoder(self, mean, logsigm, a, b, decay):
        index = torch.arange(0, self.n_topic).to(self.device)
        one_hot = F.one_hot(index, self.n_topic).to(self.device)
        mat_a = torch.ones([self.batch_size, self.n_topic, self.n_topic]).to(self.device)
        mat1 = mat_a - one_hot
        mat2 = torch.triu(mat_a)
        mat3 = torch.tril(mat_a) - one_hot
        eps0 = torch.randn([self.batch_size, self.n_hidden]).to(self.device)
        doc_vec = torch.exp(logsigm) * eps0 + mean
        eta = F.softmax(self.eta(doc_vec), 1)
        eps = torch.Tensor(self.batch_size, self.n_topic).uniform_(0.01, 0.99).to(self.device)
        v_samples = (1 - (eps ** (1 / b))) ** (1 / a)
        v_mid1 = torch.reshape(v_samples, [self.batch_size, self.n_topic, 1])
        v_mid2 = mat1 - v_mid1.repeat(1, 1, self.n_topic)
        v_mid3 = mat2 * v_mid2 + mat3
        stick_segment = -torch.prod(v_mid3, 1)
        self.remaining_stick = torch.div(stick_segment, v_samples)
        theta = stick_segment
        tempreature = 0.05 / (decay + 1e-5)
        beta_mat = torch.matmul(self.params['topic_vec'], self.params['word_vec'].T)
        beta = F.softmax(beta_mat, 1)
        depend = F.softmax(self.params['depend'] / tempreature, 1)
        theta2 = torch.matmul(theta, depend)
        tempreature = 1 ** 0.5
        beta_mat2 = torch.matmul(self.params['topic_vec2'], self.params['word_vec'].T) / tempreature
        beta2 = F.softmax(beta_mat2, 1)
        logits2 = torch.matmul(theta2, beta2)
        depend2 = F.softmax(self.params['depend2'], 1)
        theta3 = torch.matmul(theta2, depend2)
        beta_mat3 = torch.matmul(self.params['topic_vec3'], self.params['word_vec'].T)
        beta3 = F.softmax(beta_mat3, 1)
        logits3 = torch.matmul(theta3, beta3)
        logits = torch.matmul(theta, beta)
        final_logits = (torch.transpose(eta[:, 0], -1, 0) * torch.transpose(logits, 1, 0) +
                        torch.transpose(eta[:, 1], -1, 0) * torch.transpose(logits2, 1, 0) +
                        torch.transpose(eta[:, 2], -1, 0) * torch.transpose(logits3, 1, 0))
        final_logits = torch.log(final_logits.transpose(1, 0))
        return final_logits, [theta, theta2, theta3], beta, beta2, beta3

    def forward(self, x, decay):
        mean, logsigm, kld, a, b = self.encode(x)
        final_logits, theta, beta1, beta2, beta3 = self.decoder(mean, logsigm, a, b, decay)
        recons_loss = -torch.sum(final_logits * x, 1)
        return recons_loss, kld, theta, self.params['depend'], self.params['depend2'], beta1, beta2, beta3


def Beta_fn(a, b):
    return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()
        self.use_topic_represent = opt.use_topic_represent
        self.topic_num = opt.topic_num
        self.topic_attn = opt.topic_attn
        self.topic_copy = opt.topic_copy
        self.topic_attn_in = opt.topic_attn_in
        self.topic_dec = opt.topic_dec

        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout

        self.bridge = opt.bridge
        self.one2many_mode = opt.one2many_mode
        self.one2many = opt.one2many

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = opt.word2idx[pykp.io.PAD_WORD]
        self.pad_idx_trg = opt.word2idx[pykp.io.PAD_WORD]
        self.bos_idx = opt.word2idx[pykp.io.BOS_WORD]
        self.eos_idx = opt.word2idx[pykp.io.EOS_WORD]
        self.unk_idx = opt.word2idx[pykp.io.UNK_WORD]
        self.sep_idx = opt.word2idx[pykp.io.SEP_WORD]
        self.orthogonal_loss = opt.orthogonal_loss
        self.share_embeddings = opt.share_embeddings
        self.review_attn = opt.review_attn
        self.attn_mode = opt.attn_mode
        self.device = opt.device
        self.encoder = RNNEncoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )
        self.prior_mu = nn.Linear(opt.bow_vocab_size, self.decoder_size)
        self.prior_logvar = nn.Linear(opt.bow_vocab_size, self.decoder_size)

        self.posterior_pi = nn.Linear(self.decoder_size, self.topic_num)
        self.posterior_mu = nn.ModuleList(
            [nn.Linear(self.decoder_size, self.decoder_size) for _ in range(self.topic_num)])
        self.posterior_logvar = nn.ModuleList(
            [nn.Linear(self.decoder_size, self.decoder_size) for _ in range(self.topic_num)])

        self.decoder = RNNDecoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            coverage_attn=self.coverage_attn,
            copy_attn=self.copy_attn,
            review_attn=self.review_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            dropout=self.dropout,
            use_topic_represent=self.use_topic_represent,  # yue
            topic_attn=self.topic_attn,
            topic_attn_in=self.topic_attn_in,
            topic_copy=self.topic_copy,
            topic_dec=self.topic_dec,
            topic_num=self.topic_num
        )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = torch.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, \
                'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        if self.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        if not self.share_embeddings:
            self.decoder.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, beta, num_trgs=None):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param num_trgs: only effective in one2many mode 2, a list of num of targets in each batch, with len=batch_size
        :return:
        """
        batch_size, max_src_len = list(src.size())
        prior_mu = []
        prior_logvar = []
        beta = F.softmax(beta, dim=1)
        for i in range(0, beta.shape[0]):
            prior_mu.append(self.prior_mu(beta[i]).unsqueeze(0).repeat(batch_size, 1))
            prior_logvar.append(self.prior_logvar(beta[i]).unsqueeze(0).repeat(batch_size, 1))

        # Encoding
        memory_bank, encoder_final_state = self.encoder(src, src_lens)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])
        # Decoding
        h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)

        decoder_dist_all = []
        attention_dist_all = []

        if self.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float).requires_grad_()  # [batch, max_src_seq]
            coverage_all = []
        else:
            coverage = None
            coverage_all = None
        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]
        kld_loss = []
        post_mu = []
        post_logvar = []
        for i in range(self.topic_num):
            post_mu.append(self.posterior_mu[i](h_t_init))
            post_logvar.append(self.posterior_logvar[i](h_t_init))
        h_t_init = self.reparameterize(sum(post_mu) / len(post_mu), sum(post_logvar) / len(post_mu))
        kld = self.GMMkld(post_mu, post_logvar, prior_mu, prior_logvar)
        kld_loss.append(kld)
        for t in range(max_target_length):
            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next
            post_mu = []
            post_logvar = []
            for i in range(self.topic_num):
                post_mu.append(self.posterior_mu[i](h_t))
                post_logvar.append(self.posterior_logvar[i](h_t))
            # h_t = self.reparameterize(sum(post_mu) / len(post_mu), sum(post_logvar) / len(post_mu))
            # kld = self.GMMkld(post_mu, post_logvar, prior_mu, prior_logvar)
            # kld_loss.append(kld)
            decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            y_t_next = trg[:, t]  # [batch]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
        if self.coverage_attn:
            coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
            assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))
        kld_loss = sum(kld_loss) / len(kld_loss)
        return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all, kld_loss, None, None

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state

    def init_context(self, memory_bank):
        # Init by max pooling, may support other initialization later
        context, _ = memory_bank.max(dim=1)
        return context

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def GMMkld(self, posterior_mu, posterior_logvar, prior_mu, prior_logvar):
        eps = 1e-8
        prior_mu = torch.cat([prior_mu[j].unsqueeze(0) for j in range(self.topic_num)], dim=0)
        prior_logvar = torch.cat([prior_logvar[j].unsqueeze(0) for j in range(self.topic_num)], dim=0)
        posterior_mu = torch.cat([posterior_mu[j].unsqueeze(0) for j in range(self.topic_num)], dim=0)
        posterior_logvar = torch.cat([posterior_logvar[j].unsqueeze(0) for j in range(self.topic_num)], dim=0)
        kld = torch.sum((posterior_logvar - prior_logvar + 2 * eps) * 0.5 +
                        ((prior_logvar * 0.5 + eps).exp().pow(2) + (prior_mu - posterior_mu).pow(2)) / (
                                2 * (posterior_logvar * 0.5 + eps).exp().pow(2)) - 0.5)
        # kld = -0.5 * torch.sum(1 + posterior_logvar - posterior_mu.pow(2) - posterior_logvar.exp())
        return kld / self.topic_num
