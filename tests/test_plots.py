from src.plots import mt5_per_token_loss


def test_mt5_loss_divided_by_accumulation_of_each_segment():
    # steps <= 938 ran with grad_accum 4, the resumed run with grad_accum 8
    hist = [[900, 12.0], [950, 24.0]]
    assert mt5_per_token_loss(hist) == [[900, 3.0], [950, 3.0]]
