import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):

    def __init__(self, input_c, output_c, num_h=256, depth=8):
        super(MLP, self).__init__()

        self.model = nn.Sequential()
        for i in range(depth):
            if i == 0:
                self.model.add_module('linear%d' % (i + 1), torch.nn.Linear(input_c, num_h))
                self.model.add_module('relu%d' % (i + 1), torch.nn.ReLU())
            elif i != depth - 1:
                self.model.add_module('linear%d' % (i + 1), torch.nn.Linear(num_h, num_h))
                self.model.add_module('relu%d' % (i + 1), torch.nn.ReLU())
            else:
                self.model.add_module('linear%d' % (i + 1), torch.nn.Linear(num_h, output_c))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.model(x))

class MLP_res(torch.nn.Module):

    def __init__(self, input_c, output_c, num_h=256, depth=8, res_layer=[4]):
        super(MLP_res, self).__init__()

        self.model = nn.ModuleDict()
        self.res_layer = res_layer
        self.depth = depth
        for i in range(depth):
            if i == 0:
                self.model['linear%d' % (i + 1)] = torch.nn.Linear(input_c, num_h)
                self.model['relu%d' % (i + 1)]= torch.nn.ReLU()
            elif i != depth - 1:
                if i + 1 in self.res_layer:
                    self.model['linear%d' % (i + 1)] = torch.nn.Linear(num_h + input_c, num_h)
                    self.model['relu%d' % (i + 1)] = torch.nn.ReLU()
                else:
                    self.model['linear%d' % (i + 1)] = torch.nn.Linear(num_h, num_h)
                    self.model['relu%d' % (i + 1)] = torch.nn.ReLU()
            else:
                self.model['linear%d' % (i + 1)] =  torch.nn.Linear(num_h, output_c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        tmp_x = x.clone()
        for i in range(self.depth):
            if i != self.depth - 1:
                if i + 1 in self.res_layer:
                    x = torch.cat([x, tmp_x], 1)
                x = self.model['relu%d' % (i + 1)](self.model['linear%d' % (i + 1)](x))
            else:
                x = self.model['linear%d' % (i + 1)](x)

        return self.sig(x)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


""" MLP for neural implicit shapes. The code is based on https://github.com/lioryariv/idr with adaption. """


class ImplicitNetwork(torch.nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            geometric_init=True,
            bias=1.0,
            weight_norm=True,
            multires=0,
            skip_layer=[],
            cond_layer=[],
            cond_dim=69,
            dim_cond_embed=-1,
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.cond_layer = cond_layer
        self.cond_dim = cond_dim

        self.dim_cond_embed = dim_cond_embed
        if dim_cond_embed > 0:
            self.lin_p0 = torch.nn.Linear(self.cond_dim, dim_cond_embed)
            self.cond_dim = dim_cond_embed

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in self.cond_layer:
                lin = torch.nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)

    def forward(self, input, cond, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension

        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """

        n_batch, n_point, n_dim = input.shape

        if n_batch * n_point == 0:
            return input

        # reshape to [N,?]
        input = input.reshape(n_batch * n_point, n_dim)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        if len(self.cond_layer):
            n_batch, n_cond = cond.shape
            input_cond = cond.unsqueeze(1).expand(n_batch, n_point, n_cond)
            input_cond = input_cond.reshape(n_batch * n_point, n_cond)

            if mask is not None:
                input_cond = input_cond[mask]

            if self.dim_cond_embed > 0:
                input_cond = self.lin_p0(input_cond)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch * n_point, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        return x_full.reshape(n_batch, n_point, -1)


class ConditionNetwork(torch.nn.Module):
    def __init__(
            self,
            x_mean,
            d_in,
            d_out,
            d_k,
            width,
            depth,
            d_latent=10,
            geometric_init=True,
            bias=1.0,
            weight_norm=True,
            learnable_mean=False,
            multires=0,
            skip_layer=[],
    ):
        super().__init__()

        dims = [d_in + d_latent] + [width] * depth + [d_k]
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)
        self.softmax = torch.nn.Softmax(dim=0)

        if not learnable_mean:
            self.x_mean = nn.Parameter(x_mean, requires_grad=False)
        else:
            self.x_mean = nn.Parameter(x_mean, requires_grad=True)

        assert self.x_mean.shape[1] == d_out
        self.point_num = self.x_mean.shape[0]
        self.static_code = nn.Parameter(torch.zeros(d_k, d_out))
        self.latent_code = nn.Parameter(torch.zeros(self.point_num, d_latent))

    def forward(self, input, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension

        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """

        n_batch, n_dim = input.shape

        input = torch.cat([input.unsqueeze(1).expand(-1, self.point_num, -1), self.latent_code.unsqueeze(0).expand(n_batch, -1, -1)], 2)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch, n_point, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        out = self.x_mean + torch.matmul(x_full, self.softmax(self.static_code))

        return out, x_full

class SimpleNetwork(torch.nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            d_latent=10,
            geometric_init=True,
            bias=1.0,
            weight_norm=True,
            multires=0,
            skip_layer=[],
    ):
        super().__init__()

        dims = [d_in + d_latent] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)

        self.point_num = 49281
        self.latent_code = nn.Parameter(torch.zeros(self.point_num, d_latent))

    def forward(self, input, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension

        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """

        n_batch, n_dim = input.shape

        input = torch.cat([input.unsqueeze(1).expand(-1, self.point_num, -1), self.latent_code.unsqueeze(0).expand(n_batch, -1, -1)], 2)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch, n_point, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        return x_full, self.latent_code


if __name__ == '__main__':
    # x_mean = torch.zeros(49281, 16)
    #
    # net = ConditionNetwork(x_mean, 58, 16, 10, 128, 5)
    # net2 = SimpleNetwork(58, 16,  128, 5)
    #
    # input = torch.zeros(1, 58)
    # out = net(input)
    # out2 = net2(input)
    # print(out.shape)
    # print(out2.shape)

    renderer = MLP_res(6, 3)
    print(renderer(torch.rand(100,6)).shape)