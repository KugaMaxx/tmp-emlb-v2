#include "double_window_filter.hpp"

namespace kdv {

    class DoubleWindowFilter : public edn::DoubleWindowFilter, public dv::ModuleBase {
    public:
        static const char *initDescription() {
            return "Double Window Filter.";
        };

        static void initInputs(dv::InputDefinitionList &in) {
            in.addEventInput("events");
        };

        static void initOutputs(dv::OutputDefinitionList &out) {
            out.addEventOutput("events");
        };

        static void initConfigOptions(dv::RuntimeConfig &config) {
            config.add("squareR", dv::ConfigOption::intOption("Max L1 distance.", 10, 1, 1000));
            config.add("threshold", dv::ConfigOption::intOption("Threshold value (max of window size).", 1, 1, 8));

            config.setPriorityOptions({"squareR", "threshold"});
        };

        DoubleWindowFilter() {
            sizeX    = inputs.getEventInput("events").sizeX();
            sizeY    = inputs.getEventInput("events").sizeY();
            _LENGTH_ = sizeX * sizeY;
            outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
        };

        ~DoubleWindowFilter() {}

        void run() override {
            auto inEvent  = inputs.getEventInput("events").events();
            auto outEvent = outputs.getEventOutput("events").events();

            if (!inEvent) {
                return;
            }

            for (auto &evt : inEvent) {
                bool isNoise = calculateDensity(evt.x(), evt.y(), evt.timestamp(), evt.polarity());

                if (isNoise) {
                    outEvent << evt;
                }
            }
            outEvent << dv::commit;
        };

        void configUpdate() override {
            squareR   = config.getInt("squareR");
            threshold = config.getInt("threshold");
        };
    };

}

registerModuleClass(kdv::DoubleWindowFilter)
