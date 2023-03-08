def sharpness_sigma(
        model, input_tensor, label_tensor, training_accuracy, upper=5., lower=0., search_depth=20, mtc_iter=3,
        ascent_step=3, num_batch=10, deviat_eps=1e-2, bound_eps=5e-3,
        target_loss=0.1):
    learning_rate = 1e-4
    model.eval()
    # checkpoint = torch.load(checkpoint_directory)
    # model.load_state_dict(checkpoint['model_state_dict'])

    original_weight = [v.data.clone() for k, v in model.named_parameters()]
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def get_gaussian_noise_feed_dict(perturb_ph, m):
        return {}  # PyTorch doesn't need a feed_dict

    def restore_original_weight():
        for w, v in zip(model.parameters(), original_weight):
            w.data.copy_(v)

    def add_noise_to_weights(m):
        for w in model.parameters():
            noise = torch.randn_like(w) * m
            w.data.add_(noise)

    def get_norm(m):
        norm = 0
        for w, v in zip(model.parameters(), original_weight):
            norm += ((w - v) ** 2).sum()
        norm = torch.sqrt(norm)
        return norm.item()

    def project_weights(m, norm):
        target_norm = m
        for w, v in zip(model.parameters(), original_weight):
            w.data.copy_(v + (w - v) * target_norm / norm)

    h, l = upper, lower
    for j in range(search_depth):
        m = (h + l) / 2.
        max_loss = -1.
        print(f'\nsearch_depth--- {j} m={m}')
        for i in range(mtc_iter):
            print(f'\tmtc iter--- {i} m={m}')
            restore_original_weight()
            add_noise_to_weights(m)
            for k in range(ascent_step):
                print(f'\t\tascent step: {k}')
                optimizer.zero_grad()
                outputs = model(**input_tensor)
                logits = outputs[1]
                loss = -criterion(logits, label_tensor)
                loss.backward()
                optimizer.step()
                norm = get_norm(m)
                print(f'\t\tnorm={norm}, m={m}')
                if norm > m:
                    project_weights(m, norm)
                if j % 10 == 0:
                    estimates = []
                    for _ in range(num_batch):
                        outputs = model(**input_tensor)
                        logits = outputs[1]
                        pred = logits.argmax(dim=1, keepdim=True)
                        correct = pred.eq(label_tensor.view_as(pred)).sum().item()
                        accuracy = 100. * correct / label_tensor.shape[0]
                        print(f'\t\t\t\t got acc = {accuracy}')
                        loss = criterion(logits, label_tensor)
                        print(f'\t\t\t\t loss.item() = {loss.item()}')
                        estimates.append(loss.item())
                    max_loss = max(max_loss, np.mean(estimates))
        print(f'max_loss = `{max_loss}`')
        deviate = abs(max_loss - target_loss)
        print(f'deviate got = `{deviate}`, h = `{h}`, l = `{l}`, (h-l) = `{h-l}`')
        if h - l < bound_eps or deviate < deviat_eps:
            return m, "Final"
        if deviate > target_loss:
            print('changing high to mid')
            h = m
        else:
            print('changing low to mid')
            l = m
    return m, "Incomplete"
