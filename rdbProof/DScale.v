From Stdlib Require Import Init.Nat Reals Rminmax Field Logic.Classical_Prop Psatz Lia.
From Coquelicot Require Import Coquelicot Hierarchy Compactness.
From mathcomp Require Import ssreflect ssrfun ssrbool eqtype ssrnat seq fintype bigop.
From Stdlib Require Import Vectors.Vector Sets.Ensembles Classes.RelationClasses.


Import VectorNotations.

Set Bullet Behavior "Strict Subproofs".
Open Scope R_scope.


Section Rn.
  Variable n : nat.
  Definition Rn := Vector.t R n.

  Definition le_Rn (x y : Rn) : Prop :=
    forall i : Fin.t n, Vector.nth x i <= Vector.nth y i.

  #[global] Instance le_Rn_PreO : PreOrder le_Rn.
  Proof.
    split.
    - intros x i; lra.
    - intros x y z Hxy Hyz i; specialize (Hxy i); specialize (Hyz i); lra.
  Qed.

  Definition minkowski (A B : Ensemble Rn) : Ensemble Rn :=
    fun z => exists x y, A x /\ B y /\ z = Vector.map2 Rplus x y.

  Definition smul (c : R) (A : Ensemble Rn) : Ensemble Rn :=
    fun z => exists x, A x /\ z = Vector.map (fun xi => c * xi) x.

  Definition hukuhara_diff (A B C : Ensemble Rn) : Prop :=
    minkowski B C = A.

End Rn.

Section Discrete_Budget_Scale.
  Definition Lambda := nat.
  Definition le_lambda (k₁ k₂ : Lambda) : Prop := (k₁ <= k₂)%nat.

  #[global] Instance le_lambda_PreOrder : PreOrder le_lambda.
  Proof.
    split.
    - 
      unfold le_lambda. 
      exact: leqnn.  
    - 
      unfold le_lambda. 
      intros x y z Hxy Hyz.
      exact: (leq_trans Hxy Hyz). 
  Qed.

  Definition Directed (D : Ensemble Lambda) : Prop :=
    Inhabited _ D /\
    (forall k₁ k₂, D k₁ -> D k₂ ->
       exists k₃, D k₃ /\ le_lambda k₁ k₃ /\ le_lambda k₂ k₃).

  Definition supremum (D : Ensemble Lambda) (sup : Lambda) : Prop :=
    (forall k, D k -> le_lambda k sup) /\
    (forall ub, (forall k, D k -> le_lambda k ub) -> le_lambda sup ub).

  Definition dcpo : Prop :=
    forall D, Directed D -> exists sup, supremum D sup /\ D sup.

  Lemma Lambda_infinite_directed :
    Directed (fun _ : Lambda => True).
  Proof.
    split.
    - 
      exists 0%nat.
      constructor.
    - 
      intros k₁ k₂.
      exists (maxn k₁ k₂).
      split; first exact I. 
      split. 
      + apply: leq_maxl. 
      + apply: leq_maxr. 
  Qed.

  Lemma discrete_Lambda_not_dcpo : ~ dcpo.
  Proof.
    unfold not; intros H_dcpo.
    destruct (H_dcpo (fun _ => True) Lambda_infinite_directed)
      as [sup [H_supremum _]].
    unfold supremum in H_supremum.
    destruct H_supremum as [H_is_ub _].
    assert (H_contra : le_lambda (S sup) sup).
    {
      apply H_is_ub.
      exact I.
    }
    unfold le_lambda in H_contra.
    rewrite leqNgt in H_contra.
    move: H_contra.
    rewrite ltnSn.
    by rewrite /=.
  Qed.

End Discrete_Budget_Scale.

Section Outcome_Space.
  Variable n : nat.
  Definition Outcome := Vector.t R n.

  (** Pointwise outcome order *)
  Definition le_outcome (x y : Outcome) : Prop :=
    forall i : Fin.t n, Vector.nth x i <= Vector.nth y i.
  
  #[global] Instance le_outcome_PreO : PreOrder le_outcome.
  Proof.
    split.
    - intros x i; lra.
    - intros x y z Hxy Hyz i; specialize (Hxy i); specialize (Hyz i); lra.
  Qed.

  Parameter x_ideal : Outcome.

  Parameter RawAchievable : R -> Ensemble Outcome.
  
  Hypothesis RawAchievable_nonempty :
    forall λ, exists x, RawAchievable λ x.

  Definition lower_closure (A : Ensemble Outcome) : Ensemble Outcome :=
    fun p => exists a, A a /\ le_outcome p a.

  Definition GuaranteedRegion (λ : R) : Ensemble Outcome :=
    lower_closure (RawAchievable λ).
    
  Lemma GuaranteedRegion_nonempty λ :
    exists p, GuaranteedRegion λ p.
  Proof.
    destruct (RawAchievable_nonempty λ) as [x Hx].
    exists x. exists x. split; [exact Hx| reflexivity].
  Qed.

  Definition ParetoFrontier (λ : R) : Ensemble Outcome :=
    fun x =>
      RawAchievable λ x /\
      (forall y, RawAchievable λ y /\ le_outcome y x -> le_outcome x y).

  Parameter hypervolume : Ensemble Outcome -> R.
  
  Parameter Lp_distance : R -> Outcome -> Outcome -> R.
  
  Definition Q_V (λ : R) : R := hypervolume (GuaranteedRegion λ).

  Parameter InfR : (R -> Prop) -> R.
  Axiom InfR_spec : forall P, (exists x, P x) -> 
    (forall x, P x -> InfR P <= x) /\
    (forall eps, 0 < eps -> exists x, P x /\ x < InfR P + eps).

  Definition Q_D (λ : R) (p : R) : R :=
    InfR (fun d : R =>
            exists x : Outcome,
              ParetoFrontier λ x /\ d = Lp_distance p x x_ideal).

  Axiom mono_guaranteed : 
    forall λ₁ λ₂, λ₁ <= λ₂ -> 
    Included _ (GuaranteedRegion λ₁) (GuaranteedRegion λ₂).
  
  Axiom mono_hypervolume :
    forall A B, Included _ A B -> hypervolume A <= hypervolume B.

  Section Technology.
    Variable T : Ensemble (R -> R).

    Definition implements (f : R -> R) : Prop :=
      exists ϕ, T ϕ /\ forall λ, f λ = ϕ λ.

    Definition sublinear : Prop :=
      (forall c ϕ, T ϕ -> T (fun λ => c * ϕ λ)) /\
      (forall ϕ₁ ϕ₂, T ϕ₁ -> T ϕ₂ -> T (fun λ => ϕ₁ λ + ϕ₂ λ)).

    Definition tech_hukuhara_diff (ϕ ψ : R -> R) : Prop :=
      exists γ, T γ /\ forall λ, ϕ λ = ψ λ + γ λ.

    Definition residual_efficiency (ϕ ψ : R -> R) : R -> R :=
      fun λ => InfR (fun δ => tech_hukuhara_diff ϕ (fun μ => ψ (μ + δ))).

  End Technology.

  Lemma Q_V_monotone : forall λ₁ λ₂, λ₁ <= λ₂ -> Q_V λ₁ <= Q_V λ₂.
  Proof.
    intros λ₁ λ₂ Hλ.
    unfold Q_V.
    apply mono_hypervolume.
    apply mono_guaranteed; assumption.
  Qed.

  Axiom quality_equivalence :
    forall λ, Q_V λ = hypervolume (lower_closure (ParetoFrontier λ)).

End Outcome_Space.


Section Hukuhara.
  Variable n : nat.
  Notation Outcome := (Vector.t R n).
  Notation set := (Ensemble Outcome).

  Definition norm (x : Outcome) : R :=
    sqrt (List.fold_right Rplus 0 (List.map (fun xi => xi²) (Vector.to_list x))).

  Definition minus (x y : Outcome) : Outcome :=
    Vector.map2 (fun a b => a - b) x y.

  Definition dist (x y : Outcome) : R := norm (minus x y).

  Definition set_limit_at_zero_right (F : R -> set) (L : set) : Prop :=
    forall eps : R, 0 < eps ->
      exists delta : R, 0 < delta /\
        (forall h, 0 < h < delta ->
          (forall x, L x -> exists y, F h y /\ dist x y < eps) /\
          (forall y, F h y -> exists x, L x /\ dist x y < eps)).
  
  Definition minkowski_sub (A B : set) : set :=
    fun z => exists a b, A a /\ B b /\ z = minus a b.
    
  Definition sdiv (c : R) (A : set) : set :=
    fun z => exists x, A x /\ z = Vector.map (fun xi => xi / c) x.

  Variable F : R -> set.
  
  Record Hk_differentiable (λ0 : R) : Type := mkHk {
    hk_deriv : set;  

    hk_right_lim : set_limit_at_zero_right 
      (fun h => sdiv h (minkowski_sub (F (λ0 + h)) (F λ0)))
      hk_deriv;

    hk_left_lim : set_limit_at_zero_right 
      (fun h => sdiv h (minkowski_sub (F λ0) (F (λ0 - h))))
      hk_deriv;
  }.
End Hukuhara.

Section Volume_Differentiability.

  Variable n : nat.
  Notation Outcome := (Vector.t R n).
  Notation set := (Ensemble Outcome).

  Definition minkowski_subV (A B : set) : set :=
    fun z => exists a b, A a /\ B b /\ z = Vector.map2 Rminus a b.

  Definition sdiV (c : R) (A : set) : set :=
    fun z => exists x, A x /\ z = Vector.map (fun xi => xi / c) x.
    
  Definition vec_dist (x y : Outcome) : R :=
    sqrt (List.fold_right Rplus 0 (List.map (fun xi => xi²) (Vector.to_list (Vector.map2 Rminus x y)))).

  Parameter hausdorff_distance : set -> set -> R.
  
  Variables λ_min λ_max : R.
  Hypothesis λ_bounds : λ_min < λ_max.
  Definition in_Λ (λ : R) : Prop := λ_min <= λ <= λ_max.

  Variable Guaranteed : R -> set.
  Variable hypervolume : set -> R.
  Definition V_G (λ : R) : R := hypervolume (Guaranteed λ).
  
  Record Hk_C1_Regularity (λ : R) := mk_Hk_C1 {
    D_H_G : set;
    is_hk_differentiable_right :
      forall eps : R, eps > 0 ->
      exists delta : R, delta > 0 /\
        forall h : R, 0 < h < delta ->
          in_Λ (λ + h) ->
          hausdorff_distance
            (sdiV h (minkowski_subV (Guaranteed (λ + h)) (Guaranteed λ)))
            D_H_G < eps;
    is_hk_differentiable_left :
      forall eps : R, eps > 0 ->
      exists delta : R, delta > 0 /\
        forall h : R, 0 < h < delta ->
          in_Λ (λ - h) ->
          hausdorff_distance
            (sdiV h (minkowski_subV (Guaranteed λ) (Guaranteed (λ - h))))
            D_H_G < eps
  }.

  Hypothesis GR_C1 : forall λ, in_Λ λ -> Hk_C1_Regularity λ.

  Hypothesis GR_convex : forall λ, in_Λ λ ->
    forall x y, Guaranteed λ x -> Guaranteed λ y ->
    forall θ, 0 <= θ <= 1 ->
      let z := Vector.map2 (fun xi yi => θ * xi + (1 - θ) * yi) x y in
      Guaranteed λ z.

  Definition in_Λ_dec (λ : R) : {in_Λ λ} + {~in_Λ λ}.
  Proof.
    unfold in_Λ.
    destruct (Rle_dec λ_min λ) as [Hmin | Hmin_not].
    - destruct (Rle_dec λ λ_max) as [Hmax | Hmax_not].
      + left. split; assumption.
      + right. intro H; apply Hmax_not; tauto.
    - right. intro H; apply Hmin_not; tauto.
  Defined.

  Parameter volume_derivative_functional : set -> set -> R.
  
  Axiom V_G_is_differentiable_on_Λ :
    forall λ (Hλ : in_Λ λ),
    is_derive V_G λ (volume_derivative_functional
                      (Guaranteed λ)
                      (D_H_G λ (GR_C1 λ Hλ))).

  Definition V_G_prime (λ : R) : R :=
    match in_Λ_dec λ with
    | left Hλ => volume_derivative_functional (Guaranteed λ) (D_H_G λ (GR_C1 λ Hλ))
    | right _ => 0
    end.

  Lemma V_G_prime_spec : forall λ (Hλ : in_Λ λ), is_derive V_G λ (V_G_prime λ).
  Proof.
    intros λ Hλ.
    unfold V_G_prime.
    destruct (in_Λ_dec λ) as [H_in | H_not_in].
    - apply V_G_is_differentiable_on_Λ; assumption.
    - exfalso; apply H_not_in; assumption.
  Qed.
 
  Hypothesis V_G_prime_is_continuous_on_Λ :
    forall x, in_Λ x -> continuous V_G_prime x.

  Definition marginal_rate_of_expansion := V_G_prime.
  
  
  Lemma closed_interval_compact (a b : R) : 
    a <= b -> compact (fun x => a <= x <= b).
  Proof.
    intros Hle.
    apply: compact_P3.
  Qed.
  
  Lemma continuous_compact_max (f : Stdlib.Reals.Rdefinitions.R -> Stdlib.Reals.Rdefinitions.R) (a b : Stdlib.Reals.Rdefinitions.R) :
    a <= b ->
    (forall x, a <= x <= b -> continuity_pt f x) ->
    exists x, a <= x <= b /\ (forall y, a <= y <= b -> f y <= f x).
  Proof.
    intros H_le H_cont.
    destruct (continuity_ab_maj f a b H_le H_cont) as [x [H_max H_in]].
    exists x.
    split; [exact H_in | exact H_max].
  Qed.


  Theorem Existence_of_an_Optimal_Efficiency_Point :
    exists λ_star, in_Λ λ_star /\
      (forall λ, in_Λ λ -> marginal_rate_of_expansion λ <= marginal_rate_of_expansion λ_star).
  Proof.
    assert (Hle : λ_min <= λ_max) by lra.
    assert (Hcont : forall x, λ_min <= x <= λ_max -> continuity_pt V_G_prime x).
    { intros x [Hlo Hhi].
      apply continuity_pt_filterlim.
      apply V_G_prime_is_continuous_on_Λ; split; assumption.
    }
    destruct (continuous_compact_max V_G_prime λ_min λ_max Hle Hcont)
      as [λ_star [Hbounds Hmax]].
    exists λ_star; split.
    - 
      unfold in_Λ; exact Hbounds.
    - 
      intros λ Hλ.
      apply Hmax; assumption.
  Qed.

  Lemma V_G_prime_attains_min_max :
    exists λ_min_mre λ_max_mre,
      in_Λ λ_min_mre /\ in_Λ λ_max_mre /\
      (forall λ, in_Λ λ -> V_G_prime λ_min_mre <= V_G_prime λ) /\
      (forall λ, in_Λ λ -> V_G_prime λ <= V_G_prime λ_max_mre).
  Proof.
    assert (Hle : λ_min <= λ_max) by lra.

    assert (Hcont : forall x, λ_min <= x <= λ_max ->
                   continuity_pt V_G_prime x).
    { intros x [Hlo Hhi].
      apply continuity_pt_filterlim.
      apply V_G_prime_is_continuous_on_Λ; split; assumption. }

    destruct (continuous_compact_max V_G_prime λ_min λ_max Hle Hcont)
      as [λ_max_mre [Hbounds_max Hmax]].

    assert (Hcont_neg : forall x, λ_min <= x <= λ_max ->
                         continuity_pt (fun t => - V_G_prime t) x).
    { intros x Hx.
      replace (fun t => - V_G_prime t)
        with (fun t => Ropp (V_G_prime t)) by reflexivity.
      apply continuity_pt_opp, Hcont; exact Hx. }

    destruct (continuous_compact_max (fun t => - V_G_prime t)
                                     λ_min λ_max Hle Hcont_neg)
      as [λ_min_mre [Hbounds_min Hmax_neg]].

    exists λ_min_mre, λ_max_mre.
    split; [ exact Hbounds_min | ].
    split; [ exact Hbounds_max | ].
    split.
  - 
    intros λ Hλ.
    specialize (Hmax_neg λ).
    destruct Hλ as [Hlo Hhi].
    specialize (Hmax_neg (conj Hlo Hhi)).
    lra.
  - 
    intros λ Hλ.
    specialize (Hmax λ).
    destruct Hλ as [Hlo Hhi].
    specialize (Hmax (conj Hlo Hhi)).      
    exact Hmax.
  Qed.

End Volume_Differentiability.
