"""Line visualization for DUT movement and manufacturing plan comparison."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from dataclasses import asdict

from .manufacturing_plan import ManufacturingPlan, LineSimulation, DUT, DUTStatus
from .ai_optimizer import ManufacturingPlanComparator


class LineFlowVisualizer:
    """Visualize DUT flow through manufacturing line."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig = None
        self.ax = None
        self.colors = {
            'station': '#FF6B35',  # Orange
            'conveyor': '#FFE66D', # Yellow  
            'dut_pass': '#4ECDC4', # Teal
            'dut_fail': '#FF6B6B', # Red
            'dut_processing': '#95E1D3' # Light green
        }
    
    def visualize_static_line(self, plan: ManufacturingPlan, simulation_results: Dict[str, Any]) -> plt.Figure:
        """Create static visualization of manufacturing line layout and performance."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Manufacturing Plan: {plan.plan_name}', fontsize=16, fontweight='bold')
        
        # 1. Line Layout Visualization
        self._plot_line_layout(ax1, plan, simulation_results)
        ax1.set_title('Line Layout & Station Performance')
        
        # 2. Yield by Station
        self._plot_station_yields(ax2, simulation_results)
        ax2.set_title('Yield by Station')
        
        # 3. DUT Flow Sankey-style
        self._plot_dut_flow(ax3, simulation_results)
        ax3.set_title('DUT Flow Through Line')
        
        # 4. Cost & Time Breakdown
        self._plot_cost_time_breakdown(ax4, plan, simulation_results)
        ax4.set_title('Cost & Time Analysis')
        
        plt.tight_layout()
        return fig
    
    def _plot_line_layout(self, ax, plan: ManufacturingPlan, results: Dict[str, Any]):
        """Plot manufacturing line layout with station performance."""
        ax.clear()
        
        stations = plan.stations
        station_metrics = results['station_metrics']
        
        # Calculate layout positions
        num_stations = len(stations)
        spacing = 2.0
        total_width = (num_stations - 1) * spacing
        
        # Draw stations
        for i, station in enumerate(stations):
            x = i * spacing
            y = 0
            
            # Get station performance
            metrics = station_metrics.get(station.station_id, {})
            yield_rate = metrics.get('yield_rate', 0)
            utilization = metrics.get('utilization', 0)
            
            # Color code by yield (green = high yield, red = low yield)
            color_intensity = yield_rate
            station_color = plt.cm.RdYlGn(color_intensity)
            
            # Draw station rectangle
            rect = patches.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                                   facecolor=station_color, 
                                   edgecolor='black', 
                                   linewidth=2)
            ax.add_patch(rect)
            
            # Station label
            ax.text(x, y, station.station_id, ha='center', va='center', 
                   fontweight='bold', fontsize=10)
            
            # Performance metrics below
            ax.text(x, y-0.6, f'Yield: {yield_rate:.1%}', ha='center', va='top', fontsize=8)
            ax.text(x, y-0.8, f'Util: {utilization:.1%}', ha='center', va='top', fontsize=8)
            
            # Draw conveyor connection (except for last station)
            if i < num_stations - 1:
                arrow = patches.FancyArrowPatch((x+0.4, y), ((i+1)*spacing-0.4, y),
                                              arrowstyle='->', 
                                              mutation_scale=20,
                                              color=self.colors['conveyor'],
                                              linewidth=3)
                ax.add_patch(arrow)
        
        # Configure axes
        ax.set_xlim(-0.8, total_width + 0.8)
        ax.set_ylim(-1.2, 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add colorbar for yield rate
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Yield Rate')\n    \n    def _plot_station_yields(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot yield rates for each station.\"\"\"\n        station_metrics = results['station_metrics']\n        \n        station_ids = list(station_metrics.keys())\n        yields = [station_metrics[sid]['yield_rate'] for sid in station_ids]\n        \n        bars = ax.bar(station_ids, yields, color=self.colors['station'], alpha=0.7)\n        \n        # Add value labels on bars\n        for bar, yield_val in zip(bars, yields):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                   f'{yield_val:.1%}', ha='center', va='bottom')\n        \n        ax.set_ylabel('Yield Rate')\n        ax.set_ylim(0, 1.1)\n        ax.grid(True, alpha=0.3)\n        plt.setp(ax.get_xticklabels(), rotation=45)\n    \n    def _plot_dut_flow(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot DUT flow through the line (simplified Sankey diagram).\"\"\"\n        ax.clear()\n        \n        line_metrics = results['line_metrics']\n        \n        # Flow data\n        total_generated = line_metrics['total_generated']\n        total_completed = line_metrics['total_completed']\n        total_failed = line_metrics['total_failed']\n        \n        # Create flow visualization\n        flow_data = {\n            'Generated': total_generated,\n            'Completed': total_completed,\n            'Failed': total_failed\n        }\n        \n        # Simple bar chart showing flow\n        categories = list(flow_data.keys())\n        values = list(flow_data.values())\n        colors = [self.colors['conveyor'], self.colors['dut_pass'], self.colors['dut_fail']]\n        \n        bars = ax.bar(categories, values, color=colors, alpha=0.7)\n        \n        # Add value labels\n        for bar, value in zip(bars, values):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,\n                   f'{value:,}', ha='center', va='bottom', fontweight='bold')\n        \n        ax.set_ylabel('DUT Count')\n        ax.grid(True, alpha=0.3)\n        \n        # Add overall yield as text\n        overall_yield = total_completed / total_generated if total_generated > 0 else 0\n        ax.text(0.02, 0.98, f'Overall Yield: {overall_yield:.1%}', \n               transform=ax.transAxes, fontsize=12, fontweight='bold',\n               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n    \n    def _plot_cost_time_breakdown(self, ax, plan: ManufacturingPlan, results: Dict[str, Any]):\n        \"\"\"Plot cost and time breakdown by station.\"\"\"\n        station_metrics = results['station_metrics']\n        \n        station_ids = list(station_metrics.keys())\n        costs = [station_metrics[sid]['total_cost'] for sid in station_ids]\n        times = [station_metrics[sid]['total_process_time'] for sid in station_ids]\n        \n        # Create dual-axis plot\n        ax2 = ax.twinx()\n        \n        # Cost bars\n        bars1 = ax.bar([x - 0.2 for x in range(len(station_ids))], costs, \n                      width=0.4, label='Total Cost (¥)', color=self.colors['dut_fail'], alpha=0.7)\n        \n        # Time bars\n        bars2 = ax2.bar([x + 0.2 for x in range(len(station_ids))], times, \n                       width=0.4, label='Process Time (s)', color=self.colors['dut_pass'], alpha=0.7)\n        \n        # Configure axes\n        ax.set_xlabel('Station')\n        ax.set_ylabel('Total Cost (¥)', color=self.colors['dut_fail'])\n        ax2.set_ylabel('Process Time (s)', color=self.colors['dut_pass'])\n        \n        ax.set_xticks(range(len(station_ids)))\n        ax.set_xticklabels(station_ids, rotation=45)\n        \n        # Add legends\n        ax.legend(loc='upper left')\n        ax2.legend(loc='upper right')\n        \n        ax.grid(True, alpha=0.3)\n\n\nclass PlanComparisonVisualizer:\n    \"\"\"Visualize comparison between multiple manufacturing plans.\"\"\"\n    \n    def __init__(self):\n        plt.style.use('seaborn-v0_8')\n        self.colors = sns.color_palette('husl', 8)\n    \n    def visualize_plan_comparison(self, comparison_results: Dict[str, Any]) -> plt.Figure:\n        \"\"\"Create comprehensive comparison visualization.\"\"\"\n        \n        fig = plt.figure(figsize=(20, 12))\n        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)\n        \n        # Main title\n        fig.suptitle('Manufacturing Plan Comparison Analysis', fontsize=18, fontweight='bold')\n        \n        # 1. Yield vs MVA Scatter Plot (Pareto Front)\n        ax1 = fig.add_subplot(gs[0, 0])\n        self._plot_yield_vs_mva(ax1, comparison_results)\n        \n        # 2. Performance Radar Chart\n        ax2 = fig.add_subplot(gs[0, 1], projection='polar')\n        self._plot_performance_radar(ax2, comparison_results)\n        \n        # 3. Cost Breakdown Stacked Bar\n        ax3 = fig.add_subplot(gs[0, 2])\n        self._plot_cost_breakdown(ax3, comparison_results)\n        \n        # 4. Throughput Comparison\n        ax4 = fig.add_subplot(gs[1, 0])\n        self._plot_throughput_comparison(ax4, comparison_results)\n        \n        # 5. Station Yield Heatmap\n        ax5 = fig.add_subplot(gs[1, 1])\n        self._plot_station_yield_heatmap(ax5, comparison_results)\n        \n        # 6. Bottleneck Analysis\n        ax6 = fig.add_subplot(gs[1, 2])\n        self._plot_bottleneck_analysis(ax6, comparison_results)\n        \n        # 7. Summary Table\n        ax7 = fig.add_subplot(gs[2, :])\n        self._plot_summary_table(ax7, comparison_results)\n        \n        return fig\n    \n    def _plot_yield_vs_mva(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot yield vs MVA scatter with Pareto front.\"\"\"\n        plans = results['plans']\n        pareto_optimal = results.get('pareto_optimal', [])\n        \n        yields = [p['metrics']['actual_yield'] for p in plans]\n        mvas = [p['metrics']['actual_mva'] for p in plans]\n        plan_ids = [p['plan_id'] for p in plans]\n        \n        # Regular points\n        scatter = ax.scatter(yields, mvas, c=range(len(plans)), \n                           cmap='viridis', s=100, alpha=0.7, edgecolors='black')\n        \n        # Highlight Pareto optimal points\n        for i, plan_id in enumerate(plan_ids):\n            if plan_id in pareto_optimal:\n                ax.scatter(yields[i], mvas[i], s=200, facecolors='none', \n                         edgecolors='red', linewidths=3, marker='o')\n        \n        # Labels and annotations\n        ax.set_xlabel('Yield Rate')\n        ax.set_ylabel('MVA (¥)')\n        ax.set_title('Yield vs MVA Trade-off\\n(Red circles = Pareto Optimal)')\n        \n        # Add plan ID annotations\n        for i, plan_id in enumerate(plan_ids):\n            ax.annotate(plan_id, (yields[i], mvas[i]), \n                       xytext=(5, 5), textcoords='offset points', fontsize=8)\n        \n        ax.grid(True, alpha=0.3)\n    \n    def _plot_performance_radar(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot radar chart comparing multiple performance metrics.\"\"\"\n        plans = results['plans']\n        \n        # Metrics to compare (normalized)\n        metrics = ['actual_yield', 'actual_mva', 'throughput']\n        metric_labels = ['Yield', 'MVA', 'Throughput']\n        \n        # Normalize metrics to 0-1 scale\n        normalized_data = {}\n        for metric in metrics:\n            values = [p['metrics'][metric] for p in plans if p['metrics'][metric] is not None]\n            if values:\n                min_val, max_val = min(values), max(values)\n                normalized_data[metric] = {\n                    'min': min_val, 'max': max_val,\n                    'values': [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 \n                             for v in values]\n                }\n        \n        # Plot radar chart for each plan\n        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()\n        angles += angles[:1]  # Complete the circle\n        \n        for i, plan in enumerate(plans[:3]):  # Limit to 3 plans for clarity\n            values = []\n            for metric in metrics:\n                if metric in normalized_data:\n                    plan_values = normalized_data[metric]['values']\n                    values.append(plan_values[i] if i < len(plan_values) else 0)\n                else:\n                    values.append(0)\n            \n            values += values[:1]  # Complete the circle\n            \n            ax.plot(angles, values, 'o-', linewidth=2, \n                   label=plan['plan_id'], color=self.colors[i])\n            ax.fill(angles, values, alpha=0.25, color=self.colors[i])\n        \n        ax.set_xticks(angles[:-1])\n        ax.set_xticklabels(metric_labels)\n        ax.set_ylim(0, 1)\n        ax.set_title('Performance Comparison')\n        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))\n        ax.grid(True)\n    \n    def _plot_cost_breakdown(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot cost breakdown for each plan.\"\"\"\n        plans = results['plans']\n        \n        plan_ids = [p['plan_id'] for p in plans]\n        material_costs = [p['cost_breakdown']['material_cost_per_unit'] for p in plans]\n        process_costs = [p['cost_breakdown']['process_cost_per_unit'] for p in plans]\n        \n        # Stacked bar chart\n        bars1 = ax.bar(plan_ids, material_costs, label='Material Cost', \n                      color=self.colors[0], alpha=0.7)\n        bars2 = ax.bar(plan_ids, process_costs, bottom=material_costs, \n                      label='Process Cost', color=self.colors[1], alpha=0.7)\n        \n        ax.set_ylabel('Cost per Unit (¥)')\n        ax.set_title('Cost Breakdown by Plan')\n        ax.legend()\n        plt.setp(ax.get_xticklabels(), rotation=45)\n        ax.grid(True, alpha=0.3)\n    \n    def _plot_throughput_comparison(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot throughput comparison.\"\"\"\n        plans = results['plans']\n        \n        plan_ids = [p['plan_id'] for p in plans]\n        throughputs = [p['metrics']['throughput'] for p in plans]\n        \n        bars = ax.bar(plan_ids, throughputs, color=self.colors[2], alpha=0.7)\n        \n        # Add value labels on bars\n        for bar, throughput in zip(bars, throughputs):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + max(throughputs)*0.01,\n                   f'{throughput:.0f}', ha='center', va='bottom')\n        \n        ax.set_ylabel('Throughput (UPH)')\n        ax.set_title('Throughput Comparison')\n        plt.setp(ax.get_xticklabels(), rotation=45)\n        ax.grid(True, alpha=0.3)\n    \n    def _plot_station_yield_heatmap(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot heatmap of station yields across plans.\"\"\"\n        plans = results['plans']\n        \n        # Collect all station IDs\n        all_stations = set()\n        for plan in plans:\n            all_stations.update(plan['station_performance'].keys())\n        all_stations = sorted(list(all_stations))\n        \n        # Create yield matrix\n        yield_matrix = []\n        plan_ids = []\n        \n        for plan in plans:\n            plan_yields = []\n            for station in all_stations:\n                yield_val = plan['station_performance'].get(station, {}).get('yield_rate', 0)\n                plan_yields.append(yield_val)\n            yield_matrix.append(plan_yields)\n            plan_ids.append(plan['plan_id'])\n        \n        # Create heatmap\n        im = ax.imshow(yield_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)\n        \n        # Configure axes\n        ax.set_xticks(range(len(all_stations)))\n        ax.set_xticklabels(all_stations, rotation=45)\n        ax.set_yticks(range(len(plan_ids)))\n        ax.set_yticklabels(plan_ids)\n        ax.set_title('Station Yield Heatmap')\n        \n        # Add text annotations\n        for i in range(len(plan_ids)):\n            for j in range(len(all_stations)):\n                text = ax.text(j, i, f'{yield_matrix[i][j]:.2f}',\n                             ha='center', va='center', color='black', fontsize=8)\n        \n        # Colorbar\n        cbar = plt.colorbar(im, ax=ax, shrink=0.6)\n        cbar.set_label('Yield Rate')\n    \n    def _plot_bottleneck_analysis(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot bottleneck analysis.\"\"\"\n        plans = results['plans']\n        \n        bottlenecks = [p['bottlenecks'] for p in plans]\n        bottleneck_counts = {}\n        \n        for bottleneck in bottlenecks:\n            if bottleneck:\n                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1\n        \n        if bottleneck_counts:\n            stations = list(bottleneck_counts.keys())\n            counts = list(bottleneck_counts.values())\n            \n            bars = ax.bar(stations, counts, color=self.colors[3], alpha=0.7)\n            \n            # Add value labels\n            for bar, count in zip(bars, counts):\n                height = bar.get_height()\n                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,\n                       f'{count}', ha='center', va='bottom')\n            \n            ax.set_ylabel('Frequency')\n            ax.set_title('Bottleneck Frequency')\n            plt.setp(ax.get_xticklabels(), rotation=45)\n        else:\n            ax.text(0.5, 0.5, 'No Bottlenecks Identified', \n                   ha='center', va='center', transform=ax.transAxes)\n            ax.set_title('Bottleneck Analysis')\n        \n        ax.grid(True, alpha=0.3)\n    \n    def _plot_summary_table(self, ax, results: Dict[str, Any]):\n        \"\"\"Plot summary table with key metrics.\"\"\"\n        ax.axis('off')\n        \n        plans = results['plans']\n        summary = results.get('summary', {})\n        \n        # Prepare table data\n        headers = ['Plan ID', 'Yield (%)', 'MVA (¥)', 'Throughput (UPH)', 'Total Cost/Unit (¥)']\n        table_data = []\n        \n        for plan in plans:\n            row = [\n                plan['plan_id'],\n                f\"{plan['metrics']['actual_yield']:.1%}\" if plan['metrics']['actual_yield'] else 'N/A',\n                f\"{plan['metrics']['actual_mva']:.1f}\" if plan['metrics']['actual_mva'] else 'N/A',\n                f\"{plan['metrics']['throughput']:.0f}\" if plan['metrics']['throughput'] else 'N/A',\n                f\"{plan['cost_breakdown']['total_cost_per_unit']:.2f}\"\n            ]\n            table_data.append(row)\n        \n        # Create table\n        table = ax.table(cellText=table_data, colLabels=headers, \n                        cellLoc='center', loc='center')\n        table.auto_set_font_size(False)\n        table.set_fontsize(10)\n        table.scale(1, 2)\n        \n        # Style table\n        for i in range(len(headers)):\n            table[(0, i)].set_facecolor('#4CAF50')\n            table[(0, i)].set_text_props(weight='bold', color='white')\n        \n        ax.set_title('Summary Comparison Table', fontsize=14, fontweight='bold', pad=20)\n\n\ndef create_example_visualizations():\n    \"\"\"Create example visualizations for manufacturing plans.\"\"\"\n    from .manufacturing_plan import create_sample_manufacturing_plans\n    from .ai_optimizer import run_ai_optimization_example\n    \n    # Get sample plans\n    plans = create_sample_manufacturing_plans()\n    \n    # Run simulations\n    comparator = ManufacturingPlanComparator()\n    comparison_results = comparator.compare_plans(plans)\n    \n    # Create visualizations\n    line_viz = LineFlowVisualizer()\n    plan_viz = PlanComparisonVisualizer()\n    \n    # Individual plan visualization\n    for i, plan in enumerate(plans):\n        simulation = LineSimulation(plan)\n        results = simulation.run_full_simulation()\n        \n        fig = line_viz.visualize_static_line(plan, results)\n        fig.savefig(f'plan_{plan.plan_id}_analysis.png', dpi=300, bbox_inches='tight')\n        print(f\"Saved visualization for {plan.plan_id}\")\n    \n    # Comparison visualization\n    fig = plan_viz.visualize_plan_comparison(comparison_results)\n    fig.savefig('plan_comparison_analysis.png', dpi=300, bbox_inches='tight')\n    print(\"Saved plan comparison visualization\")\n    \n    plt.show()\n\n\nif __name__ == \"__main__\":\n    create_example_visualizations()