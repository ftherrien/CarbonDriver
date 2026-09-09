"""Differentiable mappings from controlled inputs to full model inputs."""

import math

import torch


class PhysicsDomain:
    """Optimize controls only; reconstruct fixed and dependent physical features."""

    def __init__(self, labels, bounds, config, fixed_features=None):
        self.labels = list(labels)
        self.bounds = torch.as_tensor(bounds).clone()
        if self.bounds.shape != (2, len(self.labels)):
            raise ValueError("Physics bounds must have shape (2, number of inputs).")
        if not torch.isfinite(self.bounds).all() or (self.bounds[0] > self.bounds[1]).any():
            raise ValueError("Physics bounds must be finite and ordered.")
        self.fixed = dict(fixed_features or {})
        for name in ("current_density", "zero_eps_thickness"):
            if name in self.labels and config.get(name) is not None:
                self.fixed[name] = float(config[name])
        for i, name in enumerate(self.labels):
            if name != "zero_eps_thickness" and self.bounds[0, i] == self.bounds[1, i]:
                self.fixed.setdefault(name, float(self.bounds[0, i]))
        if set(self.fixed) - set(self.labels):
            raise ValueError("Fixed physics features must be model input columns.")
        if not all(math.isfinite(float(v)) for v in self.fixed.values()):
            raise ValueError("Fixed physics features must be finite.")

        # Preserve CarbonDriver's historical electrode area unless the caller
        # provides the campaign-specific value explicitly.
        self.area_cm2 = float(config.get("electrode_area_cm2", 1.85**2))
        if not math.isfinite(self.area_cm2) or self.area_cm2 <= 0:
            raise ValueError("electrode_area_cm2 must be positive and finite.")
        self.derive_thickness = (
            "zero_eps_thickness" in self.labels
            and "zero_eps_thickness" not in self.fixed
        )
        self.mass_source = config.get("physics_mass_source", "Ink mass")
        self.mass_scale = float(config.get("physics_mass_scale", 1.0))
        self.mass_units = config.get("physics_mass_units", "mg")
        self.gas = self.mass_source == "Catalyst mass loading" and {
            "Catalyst mass loading",
            "AgCu Ratio",
        }.issubset(self.labels)
        if self.derive_thickness and self.mass_source not in self.labels and not self.gas:
            raise ValueError(
                "Constrained physics optimization requires the configured mass source "
                f"'{self.mass_source}' (default: Ink mass in mg)."
            )
        if self.mass_units not in {"mg", "mg/cm^2"}:
            raise ValueError("physics_mass_units must be mg or mg/cm^2.")
        if not math.isfinite(self.mass_scale) or self.mass_scale <= 0:
            raise ValueError("physics_mass_scale must be positive and finite.")
        excluded = set(self.fixed)
        if self.derive_thickness:
            excluded.add("zero_eps_thickness")
        self.free_labels = [name for name in self.labels if name not in excluded]
        self.free_indices = [self.labels.index(name) for name in self.free_labels]
        self.free_bounds = self.bounds[:, self.free_indices]
        if not self.free_labels:
            raise ValueError("Physics optimization needs at least one manipulated input.")

    def expand(self, free_raw):
        """Preserve gradients, batch dimensions, dtype and device."""
        if free_raw.shape[-1] != len(self.free_labels):
            raise ValueError("Wrong number of free physics inputs.")
        values = dict(zip(self.free_labels, free_raw.unbind(-1)))
        template = free_raw[..., 0]
        values.update(
            {
                key: torch.full_like(template, float(value))
                for key, value in self.fixed.items()
            }
        )
        if self.derive_thickness:
            if self.gas:
                density = (
                    (1 - values["AgCu Ratio"]) * 8935.0
                    + values["AgCu Ratio"] * 10490.0
                )
                mass = values["Catalyst mass loading"]
            else:
                density = 10490.0
                mass = values[self.mass_source] * self.mass_scale
            if self.mass_units == "mg/cm^2":
                mass = mass * self.area_cm2
            values["zero_eps_thickness"] = (
                mass * 1e-6 / (density * self.area_cm2 * 1e-4)
            )
        return torch.stack([values[name] for name in self.labels], dim=-1)

    def metadata(self):
        return {
            "free_inputs": self.free_labels,
            "fixed_inputs": self.fixed,
            "derived_thickness": self.derive_thickness,
            "mass_source": "Catalyst mass loading" if self.gas else self.mass_source,
            "mass_scale": self.mass_scale,
            "mass_units": self.mass_units,
            "electrode_area_cm2": self.area_cm2,
        }
