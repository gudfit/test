import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Order.Interval.Set.Basic
import Mathlib.Order.CompleteLatticeIntervals
import Mathlib.Order.CompletePartialOrder
import Mathlib.Tactic
import Init.Data.Nat.Lemmas

noncomputable section

/-- The budget‐scale is the real interval `[0,1]`. -/
abbrev Λp := ↥(Set.Icc (0 : ℝ) 1)

namespace Λp

instance : Inhabited Λp := ⟨⟨0, by simp⟩⟩

#check (inferInstance : CompleteLattice Λp)
#check (inferInstance : CompletePartialOrder Λp)

end Λp
