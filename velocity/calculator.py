import numpy as np

class HailstoneCalculator:
    @staticmethod
    def kinetic_energy(velocity, mass=0.01):
        """Calculate kinetic energy (Joules) for hailstone
        Default mass: 10g (typical small hail)"""
        return 0.5 * mass * np.sum(velocity**2)
    
    @staticmethod
    def impact_force(velocity, contact_time=0.01):
        """Estimate impact force (Newtons)
        contact_time: Duration of impact in seconds"""
        return np.linalg.norm(velocity) / contact_time
    
    @staticmethod
    def trajectory(position, velocity, time_steps):
        """Predict future positions"""
        return position + velocity * time_steps