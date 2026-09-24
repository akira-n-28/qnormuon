# QNorMuon: Gauge-Canonical Spectral Optimization

## 1. Setting

Consideriamo il ramo lineare di un blocco SwiGLU

\[
f(x)
=
W_d
\left[
\sigma(W_gx)\odot (W_ux)
\right],
\]

con

\[
W_u,W_g\in\mathbb R^{m\times n},
\qquad
W_d\in\mathbb R^{n\times m},
\qquad m>n.
\]

Per comodità definiamo

\[
U=W_u,
\qquad
D=W_d^\top\in\mathbb R^{m\times n}.
\]

La riga \(u_i^\top\) di \(U\) e la riga \(d_i^\top\) di \(D\) appartengono allo stesso neurone intermedio.

### Lemma 1 — Exact SwiGLU gauge

Per ogni matrice diagonale positiva

\[
C=\operatorname{Diag}(c_1,\ldots,c_m),\qquad c_i>0,
\]

la trasformazione

\[
U\mapsto CU,
\qquad
D\mapsto C^{-1}D
\]

lascia invariata la funzione rappresentata dalla rete.

### Proof

Poiché

\[
(CU)x=C(Ux)
\]

e \(C\) è diagonale,

\[
a\odot Cb=C(a\odot b).
\]

In termini di \(W_d=D^\top\),

\[
W_d\mapsto W_dC^{-1}.
\]

Pertanto

\[
W_dC^{-1}
\left[
\sigma(W_gx)\odot CUx
\right]
=
W_dC^{-1}C
\left[
\sigma(W_gx)\odot Ux
\right]
=
f(x).
\]

Quindi lo spazio reale dei modelli è il quoziente

\[
\mathcal Q=
\left(
\mathbb R^{m\times n}\times
\mathbb R^{m\times n}
\right)
/(\mathbb R_+)^m.
\]

---

# 2. Canonical gauge: bilanciare up e down

Definiamo

\[
r_i=\|u_i\|_2,
\qquad
s_i=\|d_i\|_2
\]

e

\[
H=
\operatorname{Diag}
\left(
\frac{r_1}{s_1},\ldots,\frac{r_m}{s_m}
\right).
\]

Supponiamo per ora \(r_i,s_i>0\).

Definiamo la rappresentazione canonica

\[
\boxed{
\bar U=H^{-1/2}U,
\qquad
\bar D=H^{1/2}D.
}
\]

### Theorem 1 — Unique balanced representative

Per ogni neurone

\[
\|\bar u_i\|
=
\|\bar d_i\|
=
\sqrt{r_is_i}.
\]

Inoltre \((\bar U,\bar D)\) è:

1. gauge-invariant;
2. funzionalmente equivalente a \((U,D)\);
3. l’unico elemento positivo dell’orbita che soddisfa
   \[
   \|\bar u_i\|=\|\bar d_i\|
   \quad\forall i.
   \]

### Proof

Per il neurone \(i\),

\[
\|\bar u_i\|
=
\sqrt{\frac{s_i}{r_i}}r_i
=
\sqrt{r_is_i},
\]

e analogamente

\[
\|\bar d_i\|
=
\sqrt{\frac{r_i}{s_i}}s_i
=
\sqrt{r_is_i}.
\]

Sotto il gauge

\[
U'=CU,\qquad D'=C^{-1}D
\]

abbiamo

\[
r_i'=c_ir_i,\qquad
s_i'=c_i^{-1}s_i,
\]

da cui

\[
H'=C^2H.
\]

Quindi

\[
(H')^{-1/2}U'
=
C^{-1}H^{-1/2}CU
=
H^{-1/2}U
\]

e

\[
(H')^{1/2}D'
=
CH^{1/2}C^{-1}D
=
H^{1/2}D.
\]

La rappresentazione canonica è dunque identica per tutti i rappresentanti della stessa orbita.

Per ottenere norme uguali deve inoltre valere

\[
c_ir_i=c_i^{-1}s_i,
\]

quindi necessariamente

\[
c_i=\sqrt{\frac{s_i}{r_i}}.
\]

L'unicità segue immediatamente.

---

# 3. Anche il gradiente ha una rappresentazione canonica

Siano

\[
G_U=\nabla_U L,
\qquad
G_D=\nabla_D L.
\]

Sotto il gauge precedente i gradienti euclidei trasformano controvariantemente:

\[
G_U'\!=C^{-1}G_U,
\qquad
G_D'\!=CG_D.
\]

Definiamo allora

\[
\boxed{
\bar G_U=H^{1/2}G_U,
\qquad
\bar G_D=H^{-1/2}G_D.
}
\]

### Theorem 2 — Gauge-invariant optimizer state

Le due quantità

\[
\bar G_U,\qquad\bar G_D
\]

sono completamente gauge-invariant.

Di conseguenza qualsiasi stato costruito esclusivamente da esse, ad esempio

\[
M_{U,t}
=
\beta M_{U,t-1}
+(1-\beta)\bar G_{U,t},
\]

\[
M_{D,t}
=
\beta M_{D,t-1}
+(1-\beta)\bar G_{D,t},
\]

è anch'esso indipendente dalla parametrizzazione scelta.

### Consequence

Questo è più forte della semplice equivarianza dell'update.

Possiamo applicare a metà del training una riscalatura arbitraria

\[
(U,D)\rightarrow(CU,C^{-1}D)
\]

e lo stato interno di QNorMuon non deve cambiare.

La dinamica sul quoziente continua dalla stessa posizione.

---

# 4. Gauge-canonical Muon

Applichiamo ora Muon non a \(G_U,G_D\), ma alle quantità canoniche.

Sia

\[
P_U=\operatorname{polar}(M_U),
\qquad
P_D=\operatorname{polar}(M_D).
\]

L'update nelle coordinate originali viene ottenuto tramite il lift

\[
\boxed{
\Delta U=H^{1/2}P_U,
\qquad
\Delta D=H^{-1/2}P_D.
}
\]

Chiamiamo questa operazione **quotient polar**.

### Theorem 3 — Exact diagonal gauge equivariance

Sotto una qualsiasi trasformazione

\[
U'=CU,\qquad D'=C^{-1}D,
\]

QMuon soddisfa esattamente

\[
\boxed{
\Delta U'=C\Delta U,
\qquad
\Delta D'=C^{-1}\Delta D.
}
\]

### Proof

Gli stati canonici \(M_U,M_D\) sono identici in entrambe le parametrizzazioni, quindi anche \(P_U,P_D\).

Poiché

\[
(H')^{1/2}=CH^{1/2},
\]

otteniamo

\[
\Delta U'
=
(H')^{1/2}P_U
=
CH^{1/2}P_U
=
C\Delta U.
\]

Analogamente

\[
\Delta D'=C^{-1}\Delta D.
\]

Pertanto due modelli funzionalmente identici rimangono funzionalmente identici dopo ogni update.

Questa proprietà vale anche se `polar` è implementata con Newton–Schulz o Polar Express: l'input dell'algoritmo numerico è letteralmente identico nelle due gauge.

La polarizzazione non deve essere esatta per ottenere la gauge-equivariance.

---

# 5. Variational characterization

La costruzione precedente non è soltanto un cambio di coordinate.

Per \(H\succ0\) consideriamo

\[
\max_{\Delta}
\langle M,\Delta\rangle_F
\]

subject to

\[
\boxed{
\Delta^\top H^{-1}\Delta=I_n.
}
\]

Ponendo

\[
Y=H^{-1/2}\Delta
\]

il problema diventa

\[
\max_{Y^\top Y=I}
\left\langle
H^{1/2}M,Y
\right\rangle_F.
\]

### Theorem 4 — Weighted spectral steepest descent

Se \(H^{1/2}M\) ha rango colonna pieno,

\[
\boxed{
\Delta^\star
=
H^{1/2}
\operatorname{polar}(H^{1/2}M).
}
\]

Inoltre

\[
\Delta^{\star\top}H^{-1}\Delta^\star=I.
\]

Quindi tutti i generalized singular values dell'update rispetto alla metrica \(H\) valgono esattamente uno:

\[
\boxed{
\kappa_H(\Delta^\star)=1.
}
\]

Il normale Muon impone condition number uno nelle coordinate euclidee arbitrarie.

QMuon lo impone nella geometria che trasforma correttamente sotto la simmetria funzionale della rete.

---

# 6. Perché precisamente \(H_i=\|u_i\|/\|d_i\|\)?

Questo rapporto non è una scelta estetica.

È essenzialmente forzato dalla simmetria.

Consideriamo una metrica locale diagonale

\[
A_i=a(r_i,s_i)>0.
\]

Richiediamo due proprietà naturali.

### Gauge covariance

\[
a(cr,c^{-1}s)=c^2a(r,s).
\]

### Up/down duality

Scambiando i due lati della coppia, la metrica deve invertirsi:

\[
a(s,r)=a(r,s)^{-1}.
\]

### Theorem 5 — Uniqueness of the canonical metric

Sotto queste due condizioni,

\[
\boxed{
a(r,s)=\frac{r}{s}.
}
\]

### Proof

Sia

\[
p=rs.
\]

Scegliendo

\[
c=\sqrt{\frac{s}{r}}
\]

portiamo \((r,s)\) in

\[
(\sqrt p,\sqrt p).
\]

La gauge covariance implica quindi

\[
a(r,s)
=
\frac{r}{s}\psi(p)
\]

per qualche funzione positiva \(\psi\).

Dalla dualità,

\[
\frac{s}{r}\psi(p)
=
\frac{1}{
(r/s)\psi(p)
},
\]

quindi

\[
\psi(p)^2=1.
\]

Poiché \(\psi>0\),

\[
\psi(p)=1.
\]

Segue

\[
a(r,s)=r/s.
\]

Quindi la metrica usata da QMuon è l'unica metrica norm-local, gauge-covariant e simmetrica rispetto allo scambio dei due lati.

---

# 7. Il problema specifico di NorMuon

Il secondo obiettivo è evitare neuroni scarsamente utilizzati.

Ma vogliamo evitare una normalizzazione

\[
P\rightarrow SP
\]

successiva alla polarizzazione, perché in generale

\[
P^\top S^2P\neq I.
\]

Quindi l'adattività neuron-wise dovrebbe entrare **prima** della polarizzazione.

Introduciamo

\[
K(x)
=
\operatorname{Diag}(e^{x_1},\ldots,e^{x_m}),
\]

dove \(x\) è uno stato gauge-invariant condiviso dalla coppia `up/down`.

Calcoliamo

\[
P_U(x)
=
\operatorname{polar}
\left(
K(x)^{1/2}M_U
\right),
\]

\[
P_D(x)
=
\operatorname{polar}
\left(
K(x)^{1/2}M_D
\right).
\]

Poiché la polarizzazione è l'ultima operazione,

\[
P_U^\top P_U=I,
\qquad
P_D^\top P_D=I
\]

rimangono esatti.

Definiamo le leverage score

\[
\ell_i^U
=
\|P_U[i,:]\|_2^2,
\]

\[
\ell_i^D
=
\|P_D[i,:]\|_2^2.
\]

Invece di uniformizzare separatamente up e down, imponiamo

\[
\boxed{
\ell_i^U+\ell_i^D
=
\frac{2n}{m}.
}
\]

È l'energia spettrale totale associata all'unità funzionale \(i\).

---

# 8. Il balancing è un problema convesso

Definiamo

\[
\begin{aligned}
\Phi(x)
={}&
\log\det
\left(
M_U^\top K(x)M_U
\right)
\\
&+
\log\det
\left(
M_D^\top K(x)M_D
\right)
-
\frac{2n}{m}
\sum_i x_i.
\end{aligned}
\]

### Theorem 6 — Leverage scores are the gradient

Vale

\[
\boxed{
\frac{\partial\Phi}{\partial x_i}
=
\ell_i^U+\ell_i^D-\frac{2n}{m}.
}
\]

### Proof

Per una generica \(M\),

\[
A=M^\top KM.
\]

Allora

\[
\frac{\partial}{\partial x_i}\log\det A
=
e^{x_i}
m_i^\top
A^{-1}
m_i.
\]

Ma

\[
P
=
K^{1/2}M
(M^\top KM)^{-1/2},
\]

perciò

\[
\|P[i,:]\|^2
=
e^{x_i}
m_i^\top
(M^\top KM)^{-1}
m_i.
\]

Sommando i due contributi otteniamo il risultato.

---

# 9. Convexity theorem

Per Cauchy–Binet,

\[
\det(M^\top KM)
=
\sum_{|S|=n}
\det(M_S)^2
\exp
\left(
\sum_{i\in S}x_i
\right).
\]

Quindi

\[
\log\det(M^\top KM)
\]

è un `log-sum-exp` di funzioni lineari di \(x\).

### Theorem 7 — No spurious balancing minima

\[
\boxed{\Phi(x)\text{ è convessa}.}
\]

Inoltre

\[
\Phi(x+c\mathbf 1)=\Phi(x),
\]

quindi possiamo fissare

\[
\sum_i x_i=0.
\]

Se entrambe le matrici di momentum sono full-spark e \(m>n\), il target

\[
\frac{2n}{m}\mathbf1
\]

appartiene all'interno del dominio raggiungibile delle leverage score.

In questo caso il minimo esiste ed è unico modulo una costante globale.

Al minimo:

\[
\boxed{
\operatorname{diag}
(P_UP_U^\top+P_DP_D^\top)
=
\frac{2n}{m}\mathbf1.
}
\]

Abbiamo quindi simultaneamente

\[
P_U^\top P_U=I,
\]

\[
P_D^\top P_D=I,
\]

e

\[
\operatorname{diag}
(P_UP_U^\top+P_DP_D^\top)
=
\frac{2n}{m}\mathbf1.
\]

Questa è la proprietà centrale di QNorMuon:

> spectral conditioning perfetto e paired-neuron balance non sono in conflitto.

Lo diventano soltanto se si tenta di ottenere il secondo tramite una normalizzazione post-polar.

---

# 10. Online version

Non è necessario risolvere il problema convesso da zero a ogni training step.

Dato

\[
e_i
=
\ell_i^U+\ell_i^D-\frac{2n}{m},
\]

manteniamo semplicemente

\[
x_{t+1}
=
x_t-\gamma e_t
\]

oppure una versione smussata

\[
q_t
=
\beta_2q_{t-1}
+
(1-\beta_2)e_t,
\]

\[
x_{t+1}=x_t-\gamma q_t.
\]

Poi rimuoviamo la gauge globale di \(K\):

\[
x\leftarrow
x-\operatorname{mean}(x).
\]

Questo richiede soltanto \(O(m)\) memoria aggiuntiva.

L'adattatore non moltiplica l'update dopo la polarizzazione: modifica il problema spettrale del passo successivo.

---

# 11. QNorMuon algorithm

```text
Inputs:
    U = W_up                       # [m,n]
    D = W_down^T                   # [m,n]

State:
    M_u, M_d                       # canonical momentum
    x                              # shared leverage state [m]

For every step:

    # 1. Determine current gauge
    r_u = row_norm(U)
    r_d = row_norm(D)

    h = r_u / r_d

    # 2. Canonicalize gradients
    Guc = sqrt(h)[:,None] * G_u
    Gdc = rsqrt(h)[:,None] * G_d

    # These are gauge invariant.

    # 3. Canonical momentum
    M_u = beta1 * M_u + (1-beta1) * Guc
    M_d = beta1 * M_d + (1-beta1) * Gdc

    # 4. Shared intrinsic row metric
    x = x - mean(x)
    khalf = exp(0.5*x)

    # 5. Spectral updates
    P_u = polar(khalf[:,None] * M_u)
    P_d = polar(khalf[:,None] * M_d)

    # 6. Paired leverage residual
    ell = rowsum(P_u**2) + rowsum(P_d**2)
    target = 2*n/m

    q = beta2*q + (1-beta2)*(ell-target)
    x = x - gamma_balance*q
    x = x - mean(x)

    # 7. Lift from quotient coordinates
    Delta_u = sqrt(h)[:,None] * P_u
    Delta_d = rsqrt(h)[:,None] * P_d

    U -= lr * Delta_u
    D -= lr * Delta_d

    # 8. Optional but recommended:
    # move weights themselves to the canonical representative

    r_u = row_norm(U)
    r_d = row_norm(D)

    c = sqrt(r_d/r_u)

    U = c[:,None] * U
    D = rsqrt(c**2)[:,None] * D
```

L'ultimo passaggio non cambia la funzione della rete.

Mantiene semplicemente i pesi nella sezione canonica

\[
\|u_i\|=\|d_i\|
\]

ed evita che la scelta arbitraria della gauge produca valori numericamente enormi o minuscoli.

---

# 12. Exact gauge-reset property

Questa implementazione ha una proprietà utile per verificarla sperimentalmente.

Prendiamo un checkpoint e generiamo una matrice diagonale casuale

\[
C_{ii}
=
10^{z_i},
\qquad
z_i\sim U[-3,3].
\]

Costruiamo

\[
U'=CU,
\qquad
D'=C^{-1}D.
\]

I due modelli sono esattamente equivalenti.

Con QNorMuon, dopo la canonicalizzazione,

\[
\bar U'=\bar U,
\qquad
\bar D'=\bar D,
\]

e anche

\[
\bar G_U'=\bar G_U,
\qquad
\bar G_D'=\bar G_D.
\]

Pertanto, usando lo stesso minibatch e stato canonico, le due run devono produrre la stessa traiettoria funzionale, a errore floating-point vicino.

Questo fornisce un unit test molto più forte di una normale curva di validation loss.

---

# 13. Relation to existing methods

NorMuon introduce neuron-wise adaptivity dopo la spectral transformation; ciò migliora l'utilizzo dei neuroni ma la riscalatura delle righe non preserva in generale la condizione Stiefel.

Aurora affronta direttamente il problema della leverage non uniforme e ricerca update con colonne ortonormali e row norm uniformi, tramite iterazioni che alternano diagonal reweighting e polarizzazione.

DDC identifica esplicitamente il gauge continuo di SwiGLU e costruisce metodi gauge-equivariant, ma segnala che i rescaling per-channel non compongono con il Muon base perché una polarizzazione standard non commuta con quel gauge.

QNorMuon non propone quindi la gauge symmetry come novità.

La candidata novità è più specifica:

\[
\boxed{
\text{canonical gauge fixing}
+
\text{spectral LMO}
+
\text{shared paired-leverage balancing}.
}
\]

In particolare, la quotient polar fornisce una composizione esplicita tra spectral optimization e il gauge diagonale di SwiGLU.

---

# 14. Falsifiable predictions

La prima suite di esperimenti dovrebbe cercare di falsificare il metodo, non semplicemente battere una baseline.

**Gauge stress test.** Muon, NorMuon e Aurora dovrebbero cambiare traiettoria quando lo stesso modello viene riparametrizzato con \(C\) molto anisotropo. QNorMuon dovrebbe essere quasi invariato.

**Polar defect.** QNorMuon dovrebbe mantenere

\[
\|P^\top P-I\|_F
\]

al livello dell'errore del polar solver, indipendentemente dal balancing.

**Paired leverage CV.**

\[
\operatorname{CV}
\left(
\ell^U+\ell^D
\right)
\]

deve convergere verso zero.

**Separate leverage.** Non è necessario che

\[
\ell_i^U=\ell_i^D.
\]

La predizione è soltanto

\[
\ell_i^U+\ell_i^D\approx 2n/m.
\]

Questo distingue il metodo da una semplice row normalization.

**Gauge drift.** Se manteniamo la canonical section,

\[
\log\frac{\|u_i\|}{\|d_i\|}
\]

deve restare numericamente vicino a zero.

**Neuron death.** La coda bassa delle leverage score e delle activation RMS dovrebbe ridursi rispetto a Muon.

**Conditioning.** Nella metrica intrinseca:

\[
\Delta U^\top H^{-1}\Delta U\approx I,
\qquad
\Delta D^\top H\Delta D\approx I.
\]

**Ablation fondamentale.**

1. Muon.
2. NorMuon.
3. Aurora.
4. QMuon: canonical gauge, \(K=I\).
5. QNorMuon: canonical gauge + shared \(K\).
6. QNorMuon con due \(K\) separati.
7. Shared \(K\) senza gauge canonicalization.

Se (4) non migliora robustezza rispetto alle riparametrizzazioni e (5) non migliora utilizzo neuronale rispetto a (4), l'ipotesi centrale è falsa.

---

# 15. Main claim

La tesi matematica del metodo può essere riassunta così:

\[
\boxed{
\text{NorMuon normalizes coordinates;}
\quad
\text{QNorMuon normalizes equivalence classes.}
}
\]

Più formalmente, per il ramo lineare di SwiGLU QNorMuon mira a ottenere simultaneamente:

\[
\boxed{
\begin{aligned}
&\text{exact }(\mathbb R_+)^m\text{-gauge equivariance},\\
&\text{unique canonical representative},\\
&\text{generalized spectral condition number }1,\\
&\text{uniform paired leverage},\\
&\text{convex intrinsic balancing problem}.
\end{aligned}}
\]

Il punto nuovo da difendere non è quindi una nuova formula di normalizzazione.

È che **l'oggetto da ottimizzare non è \(W_{\rm up}\) o \(W_{\rm down}\) separatamente: è la loro classe di equivalenza funzionale, e la polarizzazione deve essere definita direttamente su quella classe.**
