import Mathlib.Tactic
import Mathlib.Data.Set.Lattice
import Mathlib.Logic.Function.Basic

open Set Classical
noncomputable section

universe u v

structure Setting : Type (max (u+1) (v+1)) where
  X            : Type u
  leX          : X → X → Prop
  leX_refl     : ∀ x, leX x x
  leX_antisymm : ∀ {x y}, leX x y → leX y x → x = y
  leX_trans    : ∀ {x y z}, leX x y → leX y z → leX x z
  Λ            : Type v
  Λ_nonempty   : Nonempty Λ
  leΛ          : Λ → Λ → Prop
  leΛ_refl     : ∀ l, leΛ l l
  leΛ_antisymm : ∀ {l₁ l₂}, leΛ l₁ l₂ → leΛ l₂ l₁ → l₁ = l₂
  leΛ_trans    : ∀ {l₁ l₂ l₃}, leΛ l₁ l₂ → leΛ l₂ l₃ → leΛ l₁ l₃
  sup          : Set Λ → Λ
  sup_spec     :
    ∀ {D : Set Λ},
      D.Nonempty →
      (∀ l₁ l₂, l₁ ∈ D → l₂ ∈ D →
          ∃ ub, ub ∈ D ∧ leΛ l₁ ub ∧ leΛ l₂ ub) →
      ((∀ l, l ∈ D → leΛ l (sup D)) ∧
       (∀ ub, (∀ l, l ∈ D → leΛ l ub) → leΛ (sup D) ub))

namespace Setting

instance instLE_X   (S : Setting) : LE S.X := ⟨S.leX⟩
instance instLT_X   (S : Setting) : LT S.X := ⟨fun x y => S.leX x y ∧ x ≠ y⟩

instance instLE_Λ   (S : Setting) : LE S.Λ := ⟨S.leΛ⟩
instance instLT_Λ   (S : Setting) : LT S.Λ := ⟨fun l₁ l₂ => S.leΛ l₁ l₂ ∧ l₁ ≠ l₂⟩
  
variable {S : Setting}

/-- `D` is directed in `Λ`. -/
def IsDirected (D : Set S.Λ) : Prop :=
  D.Nonempty ∧
  ∀ l₁ l₂, l₁ ∈ D → l₂ ∈ D →
    ∃ ub, ub ∈ D ∧ l₁ ≤ ub ∧ l₂ ≤ ub

/-- Lower closure of a subset of `X`. -/
def lowerClosure (A : Set S.X) : Set S.X :=
  {x | ∃ a, a ∈ A ∧ x ≤ a}

notation "lc" => lowerClosure

/-- Minimal elements of `A` (w.r.t. `≤`). -/
def minimals (A : Set S.X) : Set S.X :=
  {a | a ∈ A ∧ ∀ a', a' ∈ A → a' ≤ a → a' = a}

notation "Min(" A ")" => minimals A

end Setting

def unionIndexed {S : Setting}
    (idx : Set S.Λ) (fam : S.Λ → Set S.X) : Set S.X :=
  {x | ∃ l, l ∈ idx ∧ x ∈ fam l}

structure BICO (S : Setting.{u, v}) : Type (max (u+1) (v+1)) where
  K              : S.Λ → Set S.X → Set S.X
  A1_ext         : ∀ l (A : Set S.X), A ⊆ K l A
  A2_idem        : ∀ l (A : Set S.X), K l (K l A) = K l A
  A3_mono        : ∀ l {A B : Set S.X}, A ⊆ B → K l A ⊆ K l B
  A4_scott       :
    ∀ (A : Set S.X) {D : Set S.Λ} {lSup : S.Λ},
      S.IsDirected D →
      lSup = S.sup D →
      K lSup A = {x | ∃ l, l ∈ D ∧ x ∈ K l A}
  A5_scott_in_A  :
    ∀ l {F : Set (Set S.X)}, F.Nonempty → DirectedOn (· ⊆ ·) F →
    K l (⋃₀ F) = ⋃ A ∈ F, K l A
  A5_lambda_dir  : ∀ l₁ l₂ : S.Λ, ∃ u, l₁ ≤ u ∧ l₂ ≤ u

namespace BICO
open Setting

abbrev InformationObject (S : Setting) : Type _ := Set S.X

variable {S : Setting} (B : BICO S)

lemma monotone_in_lambda
    {A : Set S.X} {l₁ l₂ : S.Λ}
    (h₁₂ : l₁ ≤ l₂) :
    B.K l₁ A ⊆ B.K l₂ A := by
  let D : Set S.Λ := fun l => l = l₁ ∨ l = l₂

  have hDir : Setting.IsDirected (S := S) D := by
    refine And.intro ?nonempty ?directed
    · exact ⟨l₁, by left; rfl⟩
    · intro a b ha hb
      refine ⟨l₂, ?_⟩
      have hl₂ : (l₂ : S.Λ) ∈ D := by right; rfl
      refine And.intro hl₂ ?_
      have h_a : a ≤ l₂ := by
        cases ha with
        | inl h => cases h; exact h₁₂
        | inr h => cases h; exact S.leΛ_refl _
      have h_b : b ≤ l₂ := by
        cases hb with
        | inl h => cases h; exact h₁₂
        | inr h => cases h; exact S.leΛ_refl _
      exact And.intro h_a h_b

  have h_sup_spec := S.sup_spec (D := D)
                       (And.left hDir) (And.right hDir)
  have h_sup_le : S.leΛ (S.sup D) l₂ := by
    have h_upper : ∀ l, l ∈ D → S.leΛ l l₂ := by
      intro l hl
      cases hl with
      | inl h => cases h; exact h₁₂
      | inr h => cases h; exact S.leΛ_refl _
    exact (h_sup_spec.2 _ h_upper)
  have h_le_sup : S.leΛ l₂ (S.sup D) := by
    have : (l₂ : S.Λ) ∈ D := by right; rfl
    exact (h_sup_spec.1 _ this)

  have h_sup_eq : l₂ = S.sup D :=
    (S.leΛ_antisymm h_sup_le h_le_sup).symm

  have h_scott :
    B.K l₂ A = unionIndexed (S := S) D (fun l => B.K l A) :=
  B.A4_scott (A := A) (D := D) (lSup := l₂) hDir h_sup_eq

  intro x hx
  have h_in_union :
      x ∈ unionIndexed (S := S) D (fun l => B.K l A) := by
    refine ⟨l₁, ?_⟩
    exact And.intro (by left; rfl) hx
  simpa [h_scott] using h_in_union

/-! ### 2. Information objects and contexts -/

def correctedInformation (l : S.Λ)
    (A : InformationObject S) : InformationObject S :=
  B.K l A

def isContext (l : S.Λ) (C : InformationObject S) : Prop :=
  B.K l C = C

lemma univ_is_context (l : S.Λ) :
    isContext B l (Set.univ : InformationObject S) := by
  have h₁ : (B.K l (Set.univ : Set S.X)) ⊆ (Set.univ : Set S.X) := by
    intro x _; exact mem_univ _
  have h₂ : (Set.univ : Set S.X) ⊆ B.K l (Set.univ) :=
    B.A1_ext l Set.univ
  exact Set.Subset.antisymm h₁ h₂

/-! ### 3. Meet and join of a collection of contexts -/

def meetContexts (Coll : Set (InformationObject S)) : InformationObject S :=
  {x | ∀ C, C ∈ Coll → x ∈ C}

def bigUnion {S : Setting} (Coll : Set (InformationObject S)) :
    InformationObject S :=
  {x | ∃ C, C ∈ Coll ∧ x ∈ C}

def joinContexts (l : S.Λ) (Coll : Set (InformationObject S)) :
    InformationObject S :=
  B.K l (bigUnion Coll)

/-! #### 3.1  `meet` is itself a context -/

lemma meet_is_context
    {l : S.Λ} {Coll : Set (InformationObject S)}
    (hColl : ∀ C, C ∈ Coll → isContext B l C) :
    isContext B l (meetContexts (S:=S) Coll) := by   
  apply Set.Subset.antisymm
  · 
    intro x hxKC C hC
    have hKC_eq : B.K l C = C := hColl C hC
    have hIncl : meetContexts (S:=S) Coll ⊆ C := by
      intro y hy; exact hy C hC
    have hInclK : B.K l (meetContexts (S:=S) Coll) ⊆ B.K l C :=
      B.A3_mono l hIncl
    have : x ∈ B.K l C := hInclK hxKC
    simpa [hKC_eq] using this
  · 
    exact B.A1_ext l _

/-! #### 3.2  `join` is a context -/

lemma join_is_context (l : S.Λ) (Coll : Set (InformationObject S)) :
    isContext B l (joinContexts B l Coll) := by
  simpa [joinContexts] using
    B.A2_idem l (bigUnion Coll)


/-! ### 3.3 Completeness of the lattice of contexts at level `l` -/
/-- `Contexts l` is complete for inclusion. -/
theorem contexts_complete_lattice
    (l : S.Λ) (Coll : Set (InformationObject S))
    (hColl : ∀ C, C ∈ Coll → isContext B l C) :
    (∃ inf, isContext B l inf ∧
      (∀ C, C ∈ Coll → inf ⊆ C) ∧
      (∀ T, isContext B l T → (∀ C, C ∈ Coll → T ⊆ C) → T ⊆ inf)) ∧
    (∃ sup, isContext B l sup ∧
      (∀ C, C ∈ Coll → C ⊆ sup) ∧
      (∀ T, isContext B l T → (∀ C, C ∈ Coll → C ⊆ T) → sup ⊆ T)) := by
  refine And.intro ?inf_part ?sup_part

  {
    let inf := meetContexts (S:=S) Coll
    use inf
    refine ⟨?is_context, ?is_lower_bound, ?is_greatest_lower_bound⟩
    case is_context =>
      exact meet_is_context B hColl
    case is_lower_bound =>
      intro C hC x hx_in_inf
      exact hx_in_inf C hC
    case is_greatest_lower_bound =>
      intro T hT_is_context hT_is_lower_bound
      intro x hx_in_T
      intro C hC
      exact hT_is_lower_bound C hC hx_in_T
  }

  {
    let sup := joinContexts B l Coll
    use sup
    refine ⟨?is_context, ?is_upper_bound, ?is_least_upper_bound⟩
    case is_context =>
      exact join_is_context B l Coll
    case is_upper_bound =>
      intro C hC
      have h_C_subset_union : C ⊆ bigUnion Coll := fun x hx => ⟨C, hC, hx⟩
      unfold sup joinContexts
      exact Set.Subset.trans h_C_subset_union (B.A1_ext l (bigUnion Coll))

    case is_least_upper_bound =>
      intro T hT_is_context hT_is_upper_bound
      unfold sup joinContexts
      unfold isContext at hT_is_context
      rw [← hT_is_context]
      apply B.A3_mono l
      intro x hx_in_union
      rcases hx_in_union with ⟨C, hC_in_Coll, hx_in_C⟩
      exact hT_is_upper_bound C hC_in_Coll hx_in_C
  }

/-- Directed–convergence (Scott–continuity) for the family `l ↦ K l A`. -/
theorem directed_convergence
    (A : Set S.X) {D : Set S.Λ} {lStar : S.Λ}
    (hDir : Setting.IsDirected (S := S) D)
    (hSup : lStar = S.sup D) :
    (∀ l₁ l₂, l₁ ∈ D → l₂ ∈ D → l₁ ≤ l₂ → B.K l₁ A ⊆ B.K l₂ A) ∧
    (B.K lStar A = unionIndexed (S := S) D (fun l => B.K l A)) := by
  have part₁ :
      ∀ l₁ l₂, l₁ ∈ D → l₂ ∈ D → l₁ ≤ l₂ → B.K l₁ A ⊆ B.K l₂ A := by
    intro l₁ l₂ _ _ hle
    exact B.monotone_in_lambda (A := A) hle
  have part₂ :
      B.K lStar A = unionIndexed (S := S) D (fun l => B.K l A) := by
    simpa [hSup] using
      B.A4_scott (A := A) (D := D) (lSup := lStar) hDir hSup
  exact And.intro part₁ part₂

/-! ### 3.4 Some handy directed sets -/

/-- `Set.univ : Set Λ` is directed -/
lemma full_isDirected (B : BICO S) : Setting.IsDirected (S:=S) (Set.univ : Set S.Λ) := by
  refine And.intro ?inh ?dir
  · 
    rcases S.Λ_nonempty with ⟨l⟩
    exact ⟨l, mem_univ _⟩
  · 
    intro l₁ l₂ _ _
    rcases B.A5_lambda_dir l₁ l₂ with ⟨u, h₁, h₂⟩
    exact ⟨u, mem_univ _, h₁, h₂⟩

/-! ### 3.5 Plateau criteria -/

theorem plateau_criterion
    (A : Set S.X) (l0 : S.Λ)
    (hPlateau : ∀ l, l0 ≤ l → B.K l A = B.K l0 A) :
    B.K l0 A = B.K (S.sup (Set.univ : Set S.Λ)) A := by
  have hle : l0 ≤ S.sup (Set.univ : Set S.Λ) := by
    have hUpper :=
      (S.sup_spec (D := Set.univ)
        (by
          rcases S.Λ_nonempty with ⟨l⟩
          exact ⟨l, mem_univ _⟩)
        (by
          intro l₁ l₂ _ _
          rcases B.A5_lambda_dir l₁ l₂ with ⟨u, h₁, h₂⟩
          exact ⟨u, mem_univ _, h₁, h₂⟩)).1
    exact hUpper l0 (mem_univ _)
  have hEq := hPlateau _ hle 
  simp [hEq]

/-- Plateau criterion for the upward-closed set `{l | l0 ≤ l}`. -/
theorem plateau_criterion_general
    (A : Set S.X) (l0 : S.Λ)
    (hPlateau : ∀ l, l0 ≤ l → B.K l A = B.K l0 A) :
    B.K (S.sup {l | l0 ≤ l}) A = B.K l0 A := by
  let D := {l : S.Λ | l0 ≤ l}
  have hDir : IsDirected D := by
    have hNonempty : D.Nonempty := ⟨l0, S.leΛ_refl l0⟩
    have hDirProp : ∀ l₁ l₂, l₁ ∈ D → l₂ ∈ D → ∃ ub, ub ∈ D ∧ l₁ ≤ ub ∧ l₂ ≤ ub := by
      intro l₁ l₂ hl₁ hl₂
      rcases B.A5_lambda_dir l₁ l₂ with ⟨u, hu₁, hu₂⟩
      have u_in_D : u ∈ D := S.leΛ_trans hl₁ hu₁
      exact ⟨u, u_in_D, hu₁, hu₂⟩
    exact ⟨hNonempty, hDirProp⟩
  have sup_is_ub := (S.sup_spec hDir.1 hDir.2).1
  have l0_le_sup : l0 ≤ S.sup D := sup_is_ub l0 (S.leΛ_refl l0)
  exact hPlateau (S.sup D) l0_le_sup
end BICO


/-! ------------------------------------------------------------
    Section 4:  Canonical construction via lower closures
    ------------------------------------------------------------ -/
namespace CanonicalLC
open Classical Set

variable {S : Setting.{u, v}}
variable (RawAchievable : S.Λ → Set S.X)

/-- The closed, lower‐comprehensive hull of the raw region. -/
def GuaranteedRegion (l : S.Λ) : Set S.X :=
  lc (RawAchievable l)

/--  Basic properties of `lc` we need later. -/
lemma lc_monotone {A B : Set S.X} (hAB : A ⊆ B) : lc A ⊆ lc B := by
  intro x hx
  rcases hx with ⟨a, haA, hxle⟩
  exact ⟨a, hAB haA, hxle⟩

lemma lc_distrib_unionIndexed
    (idx : Set S.Λ) (fam : S.Λ → Set S.X) :
    lc (unionIndexed (S := S) idx fam) =
      unionIndexed (S := S) idx (fun l => lc (fam l)) := by
  apply Set.Subset.antisymm
  · intro x hx
    rcases hx with ⟨s, ⟨l, hlidx, hfls⟩, hxle⟩
    exact ⟨l, hlidx, ⟨s, hfls, hxle⟩⟩
  · intro x hx
    rcases hx with ⟨l, hlidx, ⟨s, hfls, hxle⟩⟩
    exact ⟨s, ⟨l, hlidx, hfls⟩, hxle⟩

/-- `GuaranteedRegion` inherits monotonicity and Scott continuity. -/
theorem guaranteedRegion_monotone_and_scott
    (S1_mono :
      ∀ {l₁ l₂ : S.Λ}, l₁ ≤ l₂ →
        RawAchievable l₁ ⊆ RawAchievable l₂)
    (S2_scott :
      ∀ (D : Set S.Λ) (lStar : S.Λ),
        Setting.IsDirected (S := S) D →
        lStar = S.sup D →
        RawAchievable lStar =
          unionIndexed (S := S) D (fun l => RawAchievable l)) :
    -- (1) monotone
    (∀ {l₁ l₂ : S.Λ}, l₁ ≤ l₂ →
        GuaranteedRegion RawAchievable l₁ ⊆
        GuaranteedRegion RawAchievable l₂)
    ∧
    -- (2) Scott-continuous
    (∀ (D : Set S.Λ) (lStar : S.Λ),
        Setting.IsDirected (S := S) D →
        lStar = S.sup D →
        GuaranteedRegion RawAchievable lStar =
          unionIndexed (S := S) D
            (fun l => GuaranteedRegion RawAchievable l)) := by
  have mono :
      ∀ {l₁ l₂ : S.Λ}, l₁ ≤ l₂ →
        GuaranteedRegion RawAchievable l₁ ⊆
        GuaranteedRegion RawAchievable l₂ := by
    intro l₁ l₂ hle
    have hIncl : RawAchievable l₁ ⊆ RawAchievable l₂ :=
      S1_mono hle
    exact lc_monotone hIncl
  have scott :
      ∀ (D : Set S.Λ) (lStar : S.Λ),
        Setting.IsDirected (S := S) D →
        lStar = S.sup D →
        GuaranteedRegion RawAchievable lStar =
          unionIndexed (S := S) D
            (fun l => GuaranteedRegion RawAchievable l) := by
    intro D lStar hDir hEq
    have hRaw :
      RawAchievable lStar =
        unionIndexed (S := S) D (fun l => RawAchievable l) :=
      S2_scott D lStar hDir hEq
    calc
      GuaranteedRegion RawAchievable lStar
          = lc (RawAchievable lStar) := rfl
      _ = lc (unionIndexed (S := S) D RawAchievable) := by
          simp [hRaw]
      _ = unionIndexed (S := S) D (fun l => lc (RawAchievable l)) := by
          simpa using
            (lc_distrib_unionIndexed (S := S) D RawAchievable)
      _ = unionIndexed (S := S) D (fun l => GuaranteedRegion RawAchievable l) := by
          rfl
  ----------------------------------------------------------------
  exact ⟨(by intro l₁ l₂ h; exact mono h),
         (by intro D lStar hD hEq; exact scott D lStar hD hEq)⟩


/-- The canonical closure operator is the union with the guaranteed region. -/
def K_can (l : S.Λ) (A : Set S.X) : Set S.X :=
  A ∪ GuaranteedRegion RawAchievable l

theorem kCan_is_fixed_point_iff_guaranteedRegion_subset
    (l : S.Λ) (C : Set S.X) :
    K_can RawAchievable l C = C ↔ GuaranteedRegion RawAchievable l ⊆ C := by
  unfold K_can
  constructor
  · intro h_eq
    intro x hxG
    have : x ∈ C ∪ GuaranteedRegion RawAchievable l := Or.inr hxG
    simpa [h_eq] using this 
  · intro h_sub
    apply Set.Subset.antisymm
    · -- C ∪ GR ⊆ C
      intro x hx
      cases hx with
      | inl hxC => exact hxC
      | inr hxG => exact h_sub hxG
    · -- C ⊆ C ∪ GR
      intro x hxC
      exact Or.inl hxC

/-- A set `A` is a lower set if for any `y` in `A`, any `x <= y` is also in `A`. -/
def IsLowerSet (A : Set S.X) : Prop :=
  ∀ ⦃x y : S.X⦄, y ∈ A → x ≤ y → x ∈ A
  
lemma lowerClosure_is_Lower_Set (M : Set S.X) :
    IsLowerSet (lc M) := by
  dsimp [IsLowerSet, lowerClosure]
  intro x y h_y h_x_le_y
  rcases h_y with ⟨a, haM, h_y_le_a⟩
  exact ⟨a, haM, S.leX_trans h_x_le_y h_y_le_a⟩

lemma guaranteedRegion_is_Lower_Set (l : S.Λ) :
    IsLowerSet (GuaranteedRegion RawAchievable l) := by
  unfold GuaranteedRegion
  apply lowerClosure_is_Lower_Set

lemma union_of_Lower_Sets_is_Lower_Set {A B : Set S.X}
    (hA : IsLowerSet A) (hB : IsLowerSet B) :
    IsLowerSet (A ∪ B) := by
  intro x y h_y h_x_le_y
  cases h_y with
  | inl hyA => exact Or.inl (hA hyA h_x_le_y)
  | inr hyB => exact Or.inr (hB hyB h_x_le_y)

lemma kCan_preserves_Lower_Sets (l : S.Λ) (A : Set S.X)
    (hA : IsLowerSet A) : IsLowerSet (K_can RawAchievable l A) := by
  unfold K_can
  apply union_of_Lower_Sets_is_Lower_Set hA (guaranteedRegion_is_Lower_Set RawAchievable l)

/-===========================================================================
  Section 4.2: Pareto Frontier
  =========================================================================== -/

/-- The Pareto frontier of a set is the set of its minimal elements. -/
def ParetoFrontier (l : S.Λ) : Set S.X :=
  Min(RawAchievable l)

section RA_props
variable
  (S1_mono :
    ∀ {l₁ l₂ : S.Λ}, l₁ ≤ l₂ →
      RawAchievable l₁ ⊆ RawAchievable l₂)
  (S2_scott :
    ∀ (D : Set S.Λ) (lStar : S.Λ),
      Setting.IsDirected (S := S) D →
      lStar = S.sup D →
      RawAchievable lStar =
        unionIndexed (S := S) D (fun l => RawAchievable l))

lemma kcan_A1_Extensivity (l : S.Λ) (A : Set S.X) :
    A ⊆ K_can RawAchievable l A := by
  intro x hx
  unfold K_can
  exact Or.inl hx

lemma kcan_A2_Idempotence (l : S.Λ) (A : Set S.X) :
    K_can RawAchievable l (K_can RawAchievable l A) =
      K_can RawAchievable l A := by
  unfold K_can
  apply Set.Subset.antisymm
  · 
    intro x hx
    cases hx with
    | inl hx₁      => exact hx₁
    | inr hxGR     => exact Or.inr hxGR
  · 
    intro x hx
    exact Or.inl hx

lemma kcan_A3_Monotone_in_A (l : S.Λ) {A B : Set S.X} (hsub : A ⊆ B) :
    K_can RawAchievable l A ⊆ K_can RawAchievable l B := by
  dsimp [K_can]
  exact Set.union_subset_union_left (GuaranteedRegion RawAchievable l) hsub

lemma kcan_A4_Scott_in_lambda
    {S : Setting} {RawAchievable : S.Λ → Set S.X}
    (S1_mono :
      ∀ {l₁ l₂ : S.Λ}, l₁ ≤ l₂ →
        RawAchievable l₁ ⊆ RawAchievable l₂)
    (S2_scott :
      ∀ (D : Set S.Λ) (lStar : S.Λ),
        Setting.IsDirected (S := S) D →
        lStar = S.sup D →
        RawAchievable lStar =
          unionIndexed (S := S) D (fun l => RawAchievable l))
    (A : Set S.X) {D : Set S.Λ} {lStar : S.Λ}
    (hDir : Setting.IsDirected (S := S) D) (hSup : lStar = S.sup D) :
    K_can RawAchievable lStar A =
      unionIndexed (S := S) D (fun l' => K_can RawAchievable l' A) := by
  unfold K_can
  rcases guaranteedRegion_monotone_and_scott RawAchievable S1_mono S2_scott
    with ⟨hGR_mono, hGR_scott⟩
  have hEqGR :
      GuaranteedRegion RawAchievable lStar =
        unionIndexed (S := S) D
          (fun l => GuaranteedRegion RawAchievable l) :=
    hGR_scott D lStar hDir hSup
  ext x
  constructor
  ·
    intro hx
    cases hx with
    | inl hxA =>
        rcases hDir.1 with ⟨l₀, hl₀⟩
        exact ⟨l₀, hl₀, Or.inl hxA⟩
    | inr hxGR =>
        rw [hEqGR] at hxGR
        rcases hxGR with ⟨l', hl', hxGRl'⟩
        exact ⟨l', hl', Or.inr hxGRl'⟩
  ·
    intro hx
    rcases hx with ⟨l', hl', hxIn⟩
    cases hxIn with
    | inl hxA =>
        exact Or.inl hxA
    | inr hxGRl' =>
        have hUpper := (S.sup_spec hDir.1 hDir.2).1
        have hl'_le_lStar : l' ≤ lStar := by
          rw [hSup]
          exact hUpper l' hl'
        have hxGRlStar : x ∈ GuaranteedRegion RawAchievable lStar :=
          hGR_mono hl'_le_lStar hxGRl'
        exact Or.inr hxGRlStar

end RA_props

section ZeroBudget
variable (zero : S.Λ)

open Set

lemma lc_empty_iff {S : Setting} (A : Set S.X) :
    lc A = (∅ : Set S.X) ↔ A = (∅ : Set S.X) := by
  constructor
  · 
    intro h_lc_empty
    apply Set.subset_empty_iff.mp
    calc
      A ⊆ lc A := by
        intro a ha
        use a
        exact ⟨ha, S.leX_refl a⟩
      _ = ∅ := h_lc_empty
  · 
    intro h_A_empty
    apply Set.eq_empty_iff_forall_notMem.mpr
    intro x hx_in_lc
    rcases hx_in_lc with ⟨a, ha, _⟩
    rw [h_A_empty] at ha
    exact ha

theorem K_zero_identity_iff_RawEmpty :
    (∀ A : Set S.X, K_can RawAchievable zero A = A) ↔ RawAchievable zero = ∅ := by
  have h_equiv_gr_empty : (∀ A, K_can RawAchievable zero A = A) ↔ GuaranteedRegion RawAchievable zero = ∅ := by
    constructor
    · intro h
      specialize h ∅
      rw [kCan_is_fixed_point_iff_guaranteedRegion_subset] at h
      exact Set.subset_empty_iff.mp h
    · intro h_gr_empty A
      rw [K_can, h_gr_empty, Set.union_empty]
  rw [h_equiv_gr_empty, GuaranteedRegion, lc_empty_iff]

end ZeroBudget

section ParetoProperties
variable (has_minimals : ∀ A : Set S.X, A.Nonempty → (Setting.minimals A).Nonempty)

theorem pareto_frontier_is_subset_of_raw_achievable (l : S.Λ) :
    ParetoFrontier RawAchievable l ⊆ RawAchievable l := by
  intro x hx
  unfold ParetoFrontier at hx
  exact hx.1

/-- Upper closure of a subset of `X`. -/
def upperClosure (A : Set S.X) : Set S.X :=
  {x | ∃ a, a ∈ A ∧ a ≤ x}

theorem upper_closure_of_pareto_equals_upper_closure_of_raw_achievable
    (l : S.Λ)
    (has_minimals : ∀ A : Set S.X, A.Nonempty → (Setting.minimals A).Nonempty) :
    upperClosure (ParetoFrontier RawAchievable l) = upperClosure (RawAchievable l) := by
  apply Set.Subset.antisymm

  · intro x hx_uc_pf
    rcases hx_uc_pf with ⟨p, hp_pf, h_p_le_x⟩
    exact ⟨p, hp_pf.1, h_p_le_x⟩
  · intro x hx_uc_ra
    rcases hx_uc_ra with ⟨a, ha_ra, h_a_le_x⟩
    let A' := {y | y ∈ RawAchievable l ∧ y ≤ a}
    have h_A'_nonempty : A'.Nonempty := ⟨a, ⟨ha_ra, S.leX_refl a⟩⟩
    have h_min_exists : (Setting.minimals A').Nonempty := has_minimals A' h_A'_nonempty
    rcases h_min_exists with ⟨p, hp_min_A'⟩
    have hp_in_PF : p ∈ ParetoFrontier RawAchievable l := by
      unfold ParetoFrontier
      refine ⟨hp_min_A'.1.1, ?_⟩
      intro y hy_ra h_y_le_p
      have y_le_a := S.leX_trans h_y_le_p hp_min_A'.1.2
      have y_in_A' : y ∈ A' := ⟨hy_ra, y_le_a⟩
      exact hp_min_A'.2 y y_in_A' h_y_le_p     
    have p_le_x := S.leX_trans hp_min_A'.1.2 h_a_le_x
    exact ⟨p, hp_in_PF, p_le_x⟩
end ParetoProperties


section LowerSetRefinement

example (l : S.Λ) (A : Set S.X) (hA_ls : IsLowerSet A) :
    IsLowerSet (K_can RawAchievable l A) :=
  kCan_preserves_Lower_Sets RawAchievable l A hA_ls

example (l : S.Λ) (A B : Set S.X)
    (_hA : IsLowerSet A) (_hB : IsLowerSet B) (hAB : A ⊆ B) :
    K_can RawAchievable l A ⊆ K_can RawAchievable l B :=
  kcan_A3_Monotone_in_A RawAchievable l hAB

theorem context_in_refined_model_is_lower_set
    (l : S.Λ) (C : Set S.X)
    (h_C_is_lower : IsLowerSet C)
    (h_C_is_fixed_point : K_can RawAchievable l C = C) :
    IsLowerSet C := by
  have h_lower_KC : IsLowerSet (K_can RawAchievable l C) :=
    kCan_preserves_Lower_Sets RawAchievable l C h_C_is_lower
  simpa [h_C_is_fixed_point] using h_lower_KC

end LowerSetRefinement

end CanonicalLC
