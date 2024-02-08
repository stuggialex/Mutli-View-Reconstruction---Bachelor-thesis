import torch

def lookup_value_at(
    pixel_coordinates: torch.Tensor, feature_map: torch.Tensor
) -> torch.Tensor:
    """Lookup the pixel at the given pixel_coordinate in the given feature_map.
    Uses Bilinear interpolation.
    Args:
        pixel_coordinates:
            Tensor of dtype torch.float32 and shape [B, N_p, 2], where N_p is
            the number of individual coordinates to look up.
            This implementation assumes the last dimension to be in the order:
                (y, x).
        feature_map:
            Image or feature map of shape [B, C, H, W].
    Returns:
        The pixelvalue of map at the given coordinates. Shape [B, N_p, C].
    """


    B, ch_, H, W = feature_map.shape



    minValues = torch.zeros_like(pixel_coordinates)
    maxValues = torch.stack([
        torch.ones(pixel_coordinates.shape[0:-1]) * (float((feature_map.shape[2])) - 2),
        torch.ones(pixel_coordinates.shape[0:-1]) * (float((feature_map.shape[3])) - 2)],
        2)

    positions = torch.max(torch.min(pixel_coordinates, maxValues), minValues)

    x = positions[:, :, 0].long()
    y = positions[:, :, 1].long()



    points00 = lookup_value_at_int(torch.stack([x, y],2),feature_map)
    points01 = lookup_value_at_int(torch.stack([x, y + torch.ones_like(y)],2),feature_map)
    points10 = lookup_value_at_int(torch.stack([x + torch.ones_like(x), y ],2),feature_map)
    points11 = lookup_value_at_int(torch.stack([x + torch.ones_like(x),y + torch.ones_like(y)],2), feature_map )



    positions00 = positions.type(torch.int32)
    fractions = torch.sub(positions, positions00.type(torch.float32))

    weightsXH = fractions[:, :, 0].unsqueeze(2).repeat(1, 1, ch_)
    weightsXL = torch.ones_like(weightsXH) - weightsXH
    weightsYH = fractions[:, :, 1].unsqueeze(2).repeat(1, 1, ch_)
    weightsYL = torch.ones_like(weightsYH) - weightsYH

    values = points00 * weightsYL * weightsXL + \
             points01 * weightsYL * weightsXH + \
             points10 * weightsYH * weightsXL + \
             points11 * weightsYH * weightsXH


    return values


def lookup_value_at_int(
    pixel_coordinates: torch.Tensor, feature_map: torch.Tensor,
) -> torch.Tensor:
    """Lookup the pixel at the given integer pixel_coordinate in the given
    feature_map.
    Args:
        pixel_coordinates:
            Tensor of dtype torch.int32 and shape [B, N_p, 2], where N_p is the
            number of individual coordinates to look up.
            This implementation assumes the last dimension to be in the order:
                (y, x).
        feature_map:
            Image or feature map of shape [B, C, H, W].
    Returns:
        The pixelvalue of map at the given coordinates. Shape [B, N_p, C].
    """
    ys = pixel_coordinates[:, :, 0]
    xs = pixel_coordinates[:, :, 1]
    batches, channels, height, width = feature_map.size()
    _, N_p, _ = pixel_coordinates.size()
    # indices [B, N_p]
    indices = ys * width + xs
    # reshape feature map for use with gather [B, C, W*H]
    feature_map = feature_map.view([batches, channels, height * width])
    # [B, 1, N_P]
    index_tensor = torch.unsqueeze(indices, 1)
    # [B, C, N_P]
    index_tensor = index_tensor.repeat([1, channels, 1])
    # do lookup [B, C, W*H], [B, C, N_P] => [B, C, N_p]
    result = torch.gather(feature_map, -1, index_tensor.long())
    result = torch.transpose(result, -2, -1)

    return result