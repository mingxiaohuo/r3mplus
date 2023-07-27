from collections import OrderedDict
def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
      load_keys = list(load_state_dict.keys())
      curr_keys = list(curr_state_dict.keys())

      redo_dp = False
      undo_dp = False
      if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
          undo_dp = True
      elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
          redo_dp = True

      if undo_dp: # this
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = k[7:]  # remove `module.`
              new_state_dict[name] = v
        # load params
      elif redo_dp:
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = 'module.' + k  # remove `module.`
              new_state_dict[name] = v
      else:
          new_state_dict = load_state_dict
      return new_state_dict
def state_dict_data_parallel_fix1(load_state_dict, curr_state_dict):
      load_keys = list(load_state_dict.keys())
      curr_keys = list(curr_state_dict.keys())

      redo_dp = False
      undo_dp = False
      if not curr_keys[0].startswith('text_model.') and load_keys[0].startswith('text_model.'):   # this
          undo_dp = True
      elif curr_keys[0].startswith('text_model.') and not load_keys[0].startswith('text_model.'):
          redo_dp = True

      if undo_dp: # this
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = k[12:]  # remove `module.`
              new_state_dict[name] = v
        # load params
      elif redo_dp:
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = 'video_model.' + k  # remove `module.`
              new_state_dict[name] = v
      else:
          new_state_dict = load_state_dict
      return new_state_dict
def state_dict_data_parallel_fix2(load_state_dict, curr_state_dict):
      load_keys = list(load_state_dict.keys())
      curr_keys = list(curr_state_dict.keys())

      redo_dp = False
      undo_dp = False
      if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
          undo_dp = True
      elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
          redo_dp = True

      if undo_dp: # this
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = k[19:]  # remove `module.`
              new_state_dict[name] = v
        # load params
      elif redo_dp:
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = 'module.video_model.' + k  # remove `module.`
              new_state_dict[name] = v
      else:
          new_state_dict = load_state_dict
      return new_state_dict
def state_dict_data_parallel_fix3(load_state_dict, curr_state_dict):
      load_keys = list(load_state_dict.keys())
      curr_keys = list(curr_state_dict.keys())

      redo_dp = False
      undo_dp = False
      if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
          undo_dp = True
      elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
          redo_dp = True

      if undo_dp: # this
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = k[15:]  # remove `module.`
              new_state_dict[name] = v
        # load params
      elif redo_dp:
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in load_state_dict.items():
              name = 'module.convnet.' + k  # remove `module.`
              new_state_dict[name] = v
      else:
          new_state_dict = load_state_dict
      return new_state_dict
def inflate_positional_embeds(curr_state_dict, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(curr_state_dict.keys())
        if 'temporal_embed' in new_state_dict and 'temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = 1
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded model has MORE frames than current...'
                          f'### loading weights, filling in the extras via')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via')
                    # if self.load_temporal_fix == 'zeros':
                    #     new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    #     new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    # elif self.load_temporal_fix in ['interp', 'bilinear']:
                    #     # interpolate
                    #     # unsqueeze so pytorch thinks its an image
                    #     mode = 'nearest'
                    #     if self.load_temporal_fix == 'bilinear':
                    #         mode = 'bilinear'
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                     (curr_num_frames, embed_dim), mode='bilinear', align_corners=True).squeeze(0)
                     
                new_state_dict['temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'pos_embed' in new_state_dict and 'pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = curr_state_dict['pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict