import numpy as np
from numpy.linalg import LinAlgError

def chol_inv(hessian, max_attempts=10):
    """
    عكس الهسيان باستخدام تحليل Cholesky مع إضافة ضجيج تدريجي للقطر إذا لم يكن موجبا قطعياً.
    نفس دالتك الأصلية — لا تغييرات في المنطق.
    """
    noise_level = 1e-4
    for i in range(max_attempts):
        try:
            L = np.linalg.cholesky(hessian)
            L_inv = np.linalg.inv(L)
            hessian_inv = L_inv.T @ L_inv
            if i > 0:
                print(f"Hessian noise adding counter: {i}")
            return hessian_inv
        except LinAlgError:
            hessian += np.eye(hessian.shape[0]) * noise_level * (i + 1)
    raise LinAlgError("Hessian matrix is not positive definite even after adding noise.")


def update_weights(w, lab, c_unl, c_unl_pred, lab_pred, mu, alpha, lam, centers, widths, eta, bias, lab_phi, c_unl_phi):
    """
    نسخة مُتّجهة (Vectorized) من تحديث الأوزان بنيوتن–رافسون.
    تطابق الصيغة النظرية الأصلية: نفس التدرّج والهسيان والانتظام L2، مع نفس عوامل
    l, u, m وتراصّ الإزاحة (bias) في متجه الأوزان.
    """

    # -------- 1) تحضير المتغيّرات (نفس منطق نسختك) --------
    # ضمّ الإزاحة إلى w في النهاية
    w_aug = np.append(w, bias)  # شكلها: (H+1, )

    # إضافة عمود الإزاحة (1) إلى phi لكلٍ من المعلّم وغير المعلّم
    # ملاحظة: ستعمل حتى لو كان lab_phi بعدده صفر صفوف (سيبقى الضرب يعطي مصفوفات أصفار صحيحة الأبعاد)
    if lab_phi is None or len(lab_phi) == 0:
        lab_phi_aug = np.zeros((0, w_aug.shape[0]))
    else:
        lab_phi_aug = np.c_[lab_phi, np.ones((lab_phi.shape[0], 1))]

    # c_unl يضم (centers + unlabeled) دائماً؛ لذا c_unl_phi موجود عادة
    if c_unl_phi is None or len(c_unl_phi) == 0:
        c_unl_phi_aug = np.zeros((0, w_aug.shape[0]))
    else:
        c_unl_phi_aug = np.c_[c_unl_phi, np.ones((c_unl_phi.shape[0], 1))]

    # أحجام مجموعتي المعلّم/غير المعلّم (كما في النسخة الأصلية)
    L = lab_phi_aug.shape[0]
    U = c_unl_phi_aug.shape[0]

    l = (1.0 / L) if L > 0 else 0.0
    u = (1.0 / U) if U > 0 else 0.0
    m = (1.0 / len(lab) + 1.0 / len(c_unl)) if len(lab) > 0 else (1.0 / len(c_unl))  # مطابقة لتعريفك

    # -------- 2) التدرّج (Gradient) — بصيغة مصفوفية --------
    # التبقيات: (p - y) للمعلّم، و (p - mu) لغير المعلّم (كما في كودك)
    if L > 0:
        y_lab = lab[:, -1]               # الأوسمة الحقيقية
        resid_lab = (lab_pred - y_lab)   # شكلها (L,)
        grad_lab = lab_phi_aug.T @ resid_lab   # شكلها (H+1,)
    else:
        grad_lab = np.zeros_like(w_aug)

    if lam != 0 and U > 0:
        resid_unl = (c_unl_pred - mu)    # (U,)
        grad_unl = c_unl_phi_aug.T @ resid_unl
    else:
        grad_unl = np.zeros_like(w_aug)

    # انتظام L2 (نفس الصيغة لديك: alpha * m * ||w_aug||^2 / 2 → مشتق = alpha * m * w_aug)
    reg_grad = (alpha * m) * w_aug

    # التدرّج الكلي كما في النسخة الأصلية
    loss_grad = l * grad_lab + lam * u * grad_unl + reg_grad

    # -------- 3) الهسيان (Hessian) — بصيغة مصفوفية مكافئة لحلقاتك --------
    # ملاحظة مهمة: في كودك، كنت تجمع:
    #   labelled_hess +=  sum_i [ p_i*(1-p_i) ] * (lab_phi_aug^T @ lab_phi_aug)
    #   unlabelled_hess += sum_i [ p_i*(1-p_i) ] * (c_unl_phi_aug^T @ c_unl_phi_aug)
    # أي أن عامل p(1-p) مجمّع كسكالار مضروب في Outer Product الإجمالي — نعيده هنا كما هو لكن مُتّجهًا.
    if L > 0:
        s_lab = np.sum(lab_pred * (1.0 - lab_pred))  # scalar
        outer_lab = lab_phi_aug.T @ lab_phi_aug      # (H+1, H+1)
        hess_lab = s_lab * outer_lab
    else:
        hess_lab = np.zeros((w_aug.shape[0], w_aug.shape[0]))

    if lam != 0 and U > 0:
        s_unl = np.sum(c_unl_pred * (1.0 - c_unl_pred))  # scalar
        outer_unl = c_unl_phi_aug.T @ c_unl_phi_aug
        hess_unl = s_unl * outer_unl
    else:
        hess_unl = np.zeros((w_aug.shape[0], w_aug.shape[0]))

    reg_hess = (alpha * m) * np.eye(w_aug.shape[0])

    hessian = l * hess_lab + lam * u * hess_unl + reg_hess

    # -------- 4) خطوة نيوتن + تحديث --------
    # استخدام Cholesky عبر دالتك chol_inv (تبقى كما هي، مع الضجيج التدريجي عند الحاجة)
    hessian_inv = chol_inv(hessian)

    step = eta * (hessian_inv @ loss_grad)     # (H+1,)
    w_aug_new = w_aug - step

    # فصل الإزاحة مرة أخرى
    bias_new = w_aug_new[-1]
    w_new = w_aug_new[:-1]

    return w_new, bias_new
