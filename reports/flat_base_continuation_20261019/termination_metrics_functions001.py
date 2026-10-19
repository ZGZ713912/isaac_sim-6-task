"""Diagnostic class candidate, copied into a reviewed trainer entry."""


class FlatTerminationMetrics:
    """Count native terminal and cause flags; causes may overlap on one frame."""
    reasons = ('fall','nonwheel_contact','knee','boundary','takeoff_timeout','closure_gap',
               'jump_request_timeout','jump_outcome_timeout','spring_travel')
    flags = ('terminated','done','success') + reasons
    windows = ('whole','first1s','after1s')
    bands = ('height_low','height_nominal','height_high')

    def __init__(self,count,device):
        import torch
        self.counts=torch.zeros(3,3,count,len(self.flags),dtype=torch.int64,device=device)

    @classmethod
    def contract(cls):
        return dict(schema='flat-native-termination-counts-v1',
            flags=list(cls.flags),windows=list(cls.windows),height_bands=list(cls.bands),
            unit='native flagged policy frames per completed optimizer update',
            step_unit='completed optimizer updates',
            source='unaltered pre-auto-reset diagnostics and reasons from the native environment',
            cause_semantics='Native cause flags can overlap; sum of causes is not a unique episode count. terminated is separate from done, success and boundary.',
            scope='Diagnostic counts only; no reward, termination, sampling, state, RNG or optimizer changes')

    def observe(self,data,behavior_valid,birth_valid):
        import torch
        height=data['commands'][:,2].detach()
        lo,hi=(.23+.305)/2,(.305+.42)/2
        height_masks=torch.stack((height<lo,(height>=lo)&(height<hi),height>=hi))
        windows=torch.stack((behavior_valid|birth_valid,birth_valid,behavior_valid))
        flags=torch.stack([data[n].detach().bool() if n in ('terminated','done','success')
                           else data['reasons'][n].detach().bool() for n in self.flags],-1)
        selected=windows[:,None,:]&height_masks[None,:,:]
        self.counts.add_(selected[:,:,:,None]&flags[None,None,:,:])

    def log(self,writer,view,selected_lanes,step):
        selected=self.counts[:,:,selected_lanes,:].sum(2).cpu().tolist()
        for w,window in enumerate(self.windows):
            for b,band in enumerate(self.bands):
                for i,flag in enumerate(self.flags):
                    writer.add_scalar(f'FlatTermination/{window}/{view}/{band}/{flag}_events',
                                      selected[w][b][i],step)

    def clear(self):
        self.counts.zero_()
