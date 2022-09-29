""" Architect controls architecture of cell by computing gradients of alphas """
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
import utils.tools as tools
from models.model import sigtemp
from torch.nn.functional import sigmoid


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, device, args, criterion, inverse_transform=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.device = device
        self.args = copy.deepcopy(args)
        self.criterion = criterion
        if type(net) == nn.parallel.DistributedDataParallel:
            self.net_in = net
            self.net = self.net_in.module
            self.v_net = copy.deepcopy(net)
        else:
            self.net = net.to(self.device)
            self.v_net = copy.deepcopy(net)
        self.w_momentum =self.args.w_momentum
        self.w_weight_decay = self.args.w_weight_decay
        if self.args.inverse:
            self.inverse_transform = inverse_transform

    def critere(self, pred, true, indice, reduction='mean'):
        if self.args.fourrier:
            weights = self.net.arch()[indice, :, :]
            weights = sigtemp(weights, self.args.temp) * self.args.sigmoid
        else:
            weights = self.net.arch[indice, :, :]
            # weights = self.net.normal_prob(self.net.arch)[indice, :, :]
            weights = sigmoid(weights) * self.args.sigmoid
        if reduction != 'mean':
            crit = nn.MSELoss(reduction=reduction)
            return (crit(pred, true) * weights).mean(dim=(-1, -2))
        else:
            crit = nn.MSELoss(reduction='none')
            return (crit(pred, true) * weights).mean()

    def virtual_step(self, trn_data, next_data, xi, w_optim, indice):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        pred = torch.zeros(trn_data[1][:, -self.args.pred_len:, :].shape).to(self.device)
        if self.args.rank == 0:
            pred, true = self._process_one_batch(trn_data, self.net)
            unreduced_loss = self.critere(pred, true, indice, reduction='none')
            try:
                gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W(), retain_graph=True)
            except RuntimeError:
                gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W(), retain_graph=True, allow_unused=True)
                for name, w in self.net.named_W():
                    try:
                        gracier = torch.autograd.grad(unreduced_loss.mean(), [w], retain_graph=True)
                    except RuntimeError:
                        print(f'{name} has no gradients!!!!!!!!˚')
            with torch.no_grad():
                for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                    m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                    vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                    w.grad = g
                for a, va in zip(self.net.A(), self.v_net.A()):
                    va.copy_(a)
        for r in range(0, self.args.world_size-1):
            if self.args.rank == r:
                pred, true = self._process_one_batch(next_data, self.v_net)
            dist.broadcast(pred.contiguous(), r)
            if self.args.rank == r+1:
                own_pred, true = self._process_one_batch(trn_data, self.net)
                unreduced_loss = self.critere(own_pred, pred, indice, reduction='none')
                gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W())
                with torch.no_grad():
                    for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                        m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                        vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                        w.grad = g
                    for a, va in zip(self.net.A(), self.v_net.A()):
                        va.copy_(a)
        return unreduced_loss

    def unrolled_backward(self, args_in, trn_data, val_data, next_data, xi, w_optim, indice):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # init config
        args = args_in
        # do virtual step (calc w`)
        unreduced_loss = self.virtual_step(trn_data, next_data, xi, w_optim, indice)
        hessian = torch.zeros(args.batch_size, args.pred_len, trn_data[1].shape[-1]).to(self.device)
        if self.args.rank == 1:
            # calc unrolled loss
            pred, true = self._process_one_batch(val_data, self.v_net)
            loss = self.criterion(pred, true)
            # compute gradient
            v_W = list(self.v_net.W())
            dw = list(torch.autograd.grad(loss, v_W))
            hessian = self.compute_hessian(dw, trn_data)
            # # clipping hessian
            # max_norm = float(args.max_hessian_grad_norm)
            # hessian_clip = copy.deepcopy(hessian)
            # for n, (h_c, h) in enumerate(zip(hessian_clip, hessian)):
            #     h_norm = torch.norm(h.detach(), dim=-1)
            #     max_coeff = h_norm / max_norm
            #     max_coeff[max_coeff < 1.0] = torch.tensor(1.0).cuda(args.gpu)
            #     hessian_clip[n] = torch.div(h, max_coeff.unsqueeze(-1))
            # check = (hessian - hessian_clip).sum(dim=(-1, -2))
            # check[check>0] = 1
            # if check.sum().item()/self.args.batch_size > 0.2:
            #     print('DANGER!!!!! TOO MUCH CLIP {}'.format(check.sum().item()/self.args.batch_size))
            # hessian = hessian_clip
        elif self.args.rank == 0:
            dw_list = []
            for i in range(self.args.batch_size):
                dw_list.append(torch.autograd.grad(unreduced_loss[i], self.net.W(), retain_graph=(i != self.args.batch_size-1)))
        dist.broadcast(hessian, 1)
        da = None
        if self.args.rank == 0:
            pred, true = self._process_one_batch(trn_data, self.v_net)
            assert pred.shape == hessian.shape
            pseudo_loss = (pred * hessian).sum()
            dw0 = torch.autograd.grad(pseudo_loss, self.v_net.W())
            if self.args.fourrier:
                weights = self.net.arch()[indice, :, :]
                d_weights = torch.zeros(self.args.batch_size, requires_grad=False)[:, None, None].to(self.device)
                for i in range(self.args.batch_size):
                    for a, b in zip(dw_list[i], dw0):
                        assert a.shape == b.shape
                        d_weights[i] += (a*b).sum()
                aux_loss = (d_weights * weights).sum()
                da = torch.autograd.grad(aux_loss, self.net.A())
                with torch.no_grad():
                    for a, d in zip(self.net.A(), da):
                        a.grad = d * xi * xi
            else:
                da = torch.zeros_like(self.net.arch).to(self.device)
                for i in range(self.args.batch_size):
                    for a, b in zip(dw_list[i], dw0):
                        da[indice[i]] += (a*b).sum()
                # update final gradient = dalpha - xi*hessian
                with torch.no_grad():
                    self.net.arch.grad = da * xi * xi
                # print(self.net.arch.grad[indice], da[indice])
        return unreduced_loss.mean(), da

    def compute_hessian(self, dw, trn_data):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01  ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        trn_data[1].requires_grad = True

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d
        pred, true = self._process_one_batch(trn_data, self.net)
        loss = self.criterion(pred, true)
        dE_pos = torch.autograd.grad(loss, [trn_data[1]])[0][:, -self.args.pred_len:, :]

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p -= 2. * eps_w * d
        pred, true = self._process_one_batch(trn_data, self.net)
        loss = self.criterion(pred, true)
        dE_neg = torch.autograd.grad(loss, [trn_data[1]])[0][:, -self.args.pred_len:, :]

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d

        hessian = (dE_pos - dE_neg) / (2. * eps_w)
        trn_data[1].requires_grad = False
        return hessian

    def _process_one_batch(self, data, model):
        batch_x = data[0].float().to(self.device)
        batch_y = data[1].float()

        batch_x_mark = data[2].float().to(self.device)
        batch_y_mark = data[3].float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = self.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
