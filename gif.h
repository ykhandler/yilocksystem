#include <jni.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <android/bitmap.h>
#include <sys/mman.h>
#include <unistd.h>
#include <android/log.h>
#include <algorithm>
#include <numeric>


#include <cstdio>
#include <string>

#include <handlerlog.h>




// Constants for LZW decoding
constexpr uint16_t LZW_STACK_SIZE = 4096;
constexpr uint16_t LZW_MAX_CODE_SIZE = 12;
constexpr uint16_t LZW_TABLE_SIZE = 4096; 

struct LZWContext {
    const uint8_t* input_start;
    const uint8_t* input_end;
    const uint8_t* input_ptr;
    
    uint8_t code_size;
    uint8_t current_code_size;
    uint16_t clear_code;
    uint16_t end_code;
    
    uint64_t current_bits;
    int bits_left;
    
    uint16_t* prefix_table;
    uint8_t* suffix_stack;
    uint8_t* stack_ptr;
    
    uint16_t next_code;
    uint16_t table_size;
    uint16_t max_code;
    
    uint16_t prev_code;
    uint16_t first_code;
};

class GifImage {
public:
    struct Frame {
        uint32_t width = 0;
        uint32_t height = 0;
        uint16_t delay = 2;
        uint32_t disposal = 1;
        std::vector<uint8_t> lzw_data;
        uint32_t color_index = 0;
        int min_code_size = 0;
        uint32_t left = 0;
        uint32_t top = 0;
        uint8_t transparent_index = 0xFF;
        uint32_t* local_color_table = nullptr;
        uint32_t color_table_size = 0;
        bool has_local_color = false;
        bool interlaced = false;  // Add interlaced flag
    };

    GifImage(uint8_t* data, size_t size) : 
        mapped_data(data),
        data_size(size),
        bg_color(0xFF000000), // Initialize to opaque black
        width(0),
        height(0),
        duration(0),
        global_color_table_size(0),
        global_color_table(nullptr),
        prev_canvas(nullptr),
        current_delay_(2),
        current_disposal_(1),
        transparent_index_(0xFF) ,
		//{
        
        loop_count_(1){
		parse_header();
        parse_frames();
		}
//    }

    ~GifImage() {
        if (prev_canvas) std::free(prev_canvas);
        if (global_color_table) std::free(global_color_table);
        close();
    }

    uint32_t get_width() const { return width; }
    uint32_t get_height() const { return height; }
    uint32_t get_frame_count() const { return frames.size(); }
    uint32_t get_duration() const { return duration; }
	uint32_t get_loop_count() const { return loop_count_; }

    // Add this method to get frame delay
    uint32_t get_frame_delay(uint32_t index) const {
        if (index >= frames.size()) return 0;
        return frames[index].delay*10;
    }


	
	
	std::string intToString(uint32_t num)
	{
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%d", num);  // Minimal size impact
    return buffer;
}

std::string intToStringHex(uint32_t num)
	{
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "0x%X", num);  // Minimal size impact
    return buffer;
}

	std::string gifGetInfos()
	{
		std::string s=std::string("Gif Details:");
		
		s=s+"\n GIF width/height: " + intToString( width)+ "x" +intToString( height) +" pixels";   // std::to_string(width);
		s=s+"\n number of frames: " + intToString( frames.size() );
		s=s+"\n number of color: " + intToString( global_color_table_size );
		s=s+"\n background color: " + intToString( bg_color );
		s=s+"\n data_size: " + intToString( data_size );
	
		s=s+"\nGlobal color table: " ;
		for(int i=0;i<global_color_table_size;i++)
		{
			s=s+"\n" + intToString(i)+": "+intToStringHex(global_color_table[i]) ;
		}
		//*/
		/*
		for(int j=0;j<frames.size()&&j<=4;j++)
		{
		Frame& frame = frames[j];
		s=s+"\nFrame #" + intToString(j+1) +":\n x:  " + intToString( frame.left );
		s=s+"\n y:  " + intToString( frame.top );
		s=s+"\n width:  " + intToString( frame.width );
		s=s+"\n height:  " + intToString( frame.height );
		s=s+"\n delay:  " + intToString( frame.delay );
		s=s+"\n transparent index:  " + intToString( frame.transparent_index );
		s=s+"\n disposal:  " + intToString( frame.disposal );
		s=s+"\n has local color:  "+ (frame.has_local_color?"true" :"false")  ;
		s=s+"\n interlaced:  "+ (frame.interlaced?"true" :"false")  ;
		}
		*/
		return s;
	}


    bool decode_frame(uint32_t index, uint8_t* output) {
    	//LOGI("decode_frame");     
        if (index >= frames.size()) return false;
        Frame& frame = frames[index];
        const size_t frame_size = frame.width * frame.height;
        
        if (frame.width == 0 || frame.height == 0 || frame_size == 0) {
            LOGE("Invalid frame dimensions");
            return false;
        }
        
        if (frame.lzw_data.empty()) {
            LOGE("Empty LZW data");
            return false;
        }
 
        
        
        LZWContext ctx;
        if (lzw_decode_init(&ctx, frame.lzw_data.data(), 
                           frame.lzw_data.size(), 
                           frame.min_code_size) != 0) {
            LOGE("LZW init failed");
            return false;
        }

        int decoded = lzw_decode(&ctx, output, frame_size);
        //LOGI("decoded: %d", decoded); 
        lzw_decode_free(&ctx);

        if (decoded < 0) {
            LOGE("LZW decoding error");
            return false;
        }
        //*/
        // Handle interlaced GIFs
        if (frame.interlaced && decoded == static_cast<int>(frame_size)) {
        	LOGI("Deinterlace");
            deinterlace(output, frame.width, frame.height);
        }
                
        return true;
    }
    
    
    
    static void deinterlace(uint8_t* data, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;
    std::vector<uint8_t> temp(data, data + width * height);
    const uint8_t* src = temp.data();
    
    const uint32_t start[] = {0, 4, 2, 1};
    const uint32_t step[] = {8, 8, 4, 2};
    const uint32_t passes = 4;
    uint32_t offset = 0;

    for (uint32_t pass = 0; pass < passes; pass++) {
        for (uint32_t y = start[pass]; y < height; y += step[pass]) {
            uint8_t* dest = data + y * width;
            memcpy(dest, src + offset, width);
            offset += width;
        }
    }
}
    
    
    
    
    void set_background_color(uint32_t color) { bg_color = color; }



static uint32_t* BuildColorTable(const uint8_t* data, uint32_t size) {
        if (!data || size == 0) return nullptr;
        uint32_t* table = static_cast<uint32_t*>(std::calloc(size * sizeof(uint32_t),sizeof(uint32_t)));
        if (!table) return nullptr;
        for (uint32_t i = 0; i < size; ++i) {

            
            table[i] = //0xFF000000 | 
                  data[3*i+2] << 16 |  // Blue -> 0x00RR0000 becomes 0x000000RR in LE
                  data[3*i+1] << 8 |   // Green
                  data[3*i];           // Red (LSB) 
		//*/
        }
        return table;
    }

    void close() {
        if (mapped_data) {
            munmap(const_cast<uint8_t*>(mapped_data), data_size);
            mapped_data = nullptr;
            data_size = 0;
        }
    }
    

    
    
    // Add this helper method to get background color
    uint32_t get_background_color() const { return bg_color; }




    bool render_frame(uint32_t frame_index, uint32_t* output, 
                     uint32_t canvas_width, uint32_t canvas_height) {
         //  	frame_index=0;
        if (frame_index >= frames.size()) return false;
        Frame& frame = frames[frame_index];

        LOGI("canvas size: %d x %d", canvas_width, canvas_height);
        LOGI("Frame size: %d x %d", frame.width, frame.height);
        LOGI("Frame position: %d x %d", frame.left, frame.top);
        // In render_frame
if (frame.left >= canvas_width || frame.top >= canvas_height) {
    LOGE("Frame position out of bounds: %d,%d (canvas: %dx%d)", 
         frame.left, frame.top, canvas_width, canvas_height);
    return false;
}

        // Handle disposal from previous frame
   handle_disposal(frame_index, output, canvas_width, canvas_height);

   // Save state BEFORE rendering current frame (for disposal method 3)
        if (frame.disposal == 3) {
            size_t size = canvas_width * canvas_height;
            saved_before_frame = std::make_unique<uint32_t[]>(size);
            if (saved_before_frame) {
                memcpy(saved_before_frame.get(), output, size * sizeof(uint32_t));
            } else {
                LOGE("Failed to allocate saved_before_frame");
            }
        }

    
        
        uint32_t* color_table = frame.local_color_table ? 
                      frame.local_color_table : 
                      global_color_table;
		uint32_t color_table_size = frame.local_color_table ? 
                          frame.color_table_size : 
                          global_color_table_size;


		if (!color_table) {
        LOGE("No color table available");
        return false;
    }
    

        // Allocate temporary buffer for decoded frame
        std::vector<uint8_t> indexed(frame.width * frame.height);
        if (!decode_frame(frame_index, indexed.data())) {
            return false;
        }

        // Convert indexed colors to ARGB and apply transparency
    uint32_t* canvas_row = output + (frame.top * canvas_width);

    
    
    for (uint32_t y = 0; y < frame.height; y++) {
        const uint32_t canvas_y = frame.top + y;
        if (canvas_y >= canvas_height) continue;
        
        uint32_t* row_ptr = output + canvas_y * canvas_width;
        const uint8_t* idx_row = indexed.data() + y * frame.width;
        
        for (uint32_t x = 0; x < frame.width; x++) {
            const uint32_t canvas_x = frame.left + x ;
            if (canvas_x >= canvas_width) continue;
            
            const uint8_t idx = idx_row[x];
            if (idx == frame.transparent_index) continue;
            
            if (idx >= color_table_size) {
                continue;
            }
            
            if (idx < color_table_size) {
                row_ptr[canvas_x] = color_table[idx];
            }
            
            
        }
    }

//*
// After rendering, if the current frame's disposal method is 1, save the state for future use.
if (frame.disposal == 1) {
    save_current_state(output, canvas_width, canvas_height);
}
//*/

        
        return true;
    }

    bool render_frame_scaled(uint32_t frame_index, uint32_t* output, 
                            uint32_t output_width, uint32_t output_height,
                            uint32_t scale_factor) {
                            	return false;
        if (frame_index >= frames.size() || scale_factor == 0) 
            return false;

        Frame& frame = frames[frame_index];
        
        // Allocate temporary buffer for decoded frame
        std::vector<uint8_t> indexed(frame.width * frame.height);
        if (!decode_frame(frame_index, indexed.data())) {
            return false;
        }

        uint32_t* color_table = frame.local_color_table ? 
                              frame.local_color_table : 
                              global_color_table;

        // Calculate scale ratios
        float x_ratio = static_cast<float>(frame.width) / 
                       static_cast<float>(output_width);
        float y_ratio = static_cast<float>(frame.height) / 
                      static_cast<float>(output_height);

        // Render with scaling
        for (uint32_t y = 0; y < output_height; y++) {
            for (uint32_t x = 0; x < output_width; x++) {
                uint32_t src_x = static_cast<uint32_t>(x * x_ratio);
                uint32_t src_y = static_cast<uint32_t>(y * y_ratio);
                
                if (src_x >= frame.width || src_y >= frame.height) 
                    continue;
                
                uint8_t idx = indexed[src_y * frame.width + src_x];
                uint32_t color = color_table[idx];
                
                if (idx == frame.transparent_index) {
                    color = bg_color;
                }
                
                output[y * output_width + x] = color;
            }
        }
        
        return true;
    }

private:
    uint8_t* mapped_data;
    size_t data_size;
    uint32_t bg_color;
    uint32_t width;
    uint32_t height;
    uint32_t duration = 0;
    uint32_t global_color_table_size = 0;
    uint32_t* global_color_table;
    std::vector<Frame> frames;
    uint32_t* prev_canvas = nullptr;
    uint32_t current_delay_;
    uint32_t current_disposal_;
    uint32_t transparent_index_;
    uint16_t loop_count_ = 1;  
   std::unique_ptr<uint32_t[]> saved_before_frame;
    

    void parse_header() {
        if (data_size < 13) {
            LOGE("Invalid GIF header");
            return;
        }
        
        // Parse width and height
        width = mapped_data[6] | (mapped_data[7] << 8);
        height = mapped_data[8] | (mapped_data[9] << 8);
        
        const uint8_t packed = mapped_data[10];
        if (packed & 0x80) {
            global_color_table_size = 1 << ((packed & 0x07) + 1);
            parse_global_color_table();
        }
        
        // Set background color from global color table
        if (global_color_table) {
            uint8_t bg_index = mapped_data[11];
            if (bg_index < global_color_table_size) {
                bg_color = global_color_table[bg_index];
            }
        }
        
        // Parse duration from header (if available)
        if (data_size > 19) {
            duration = (mapped_data[21] << 8) | 
                      mapped_data[20] | 
                      (mapped_data[22] << 16) | 
                      (mapped_data[23] << 24);
        }
    }

    void parse_global_color_table() {
        if (data_size < 13 + 3 * global_color_table_size) {
            LOGE("Invalid global color table");
            return;
        }
        global_color_table = BuildColorTable(mapped_data + 13, global_color_table_size);
    }
    

void parse_image_frame(size_t& offset) {
        if (offset + 10 >= data_size) return;
        if (mapped_data[offset] != 0x2C) return; // Image descriptor marker
        
        Frame frame;
        offset++; // Skip descriptor
        
        // Parse frame position and dimensions
        frame.left =  mapped_data[offset] | (mapped_data[offset+1] << 8);
        offset += 2;
        frame.top = mapped_data[offset] | (mapped_data[offset+1] << 8);
        offset += 2;
        frame.width = mapped_data[offset] | (mapped_data[offset+1] << 8);
        offset += 2;
        frame.height = mapped_data[offset] | (mapped_data[offset+1] << 8);
        offset += 2;
        
          
    if (frame.left >= width || frame.top >= height ||
        frame.width == 0 || frame.height == 0 ||
        frame.width > width || frame.height > height) 
    {
        LOGE("Invalid frame dimensions");
        return;
    }
        if (frame.left + frame.width > width || frame.top + frame.height > height) {
       LOGE("Frame exceeds canvas bounds");
       return;
   }
        
        // Validate frame dimensions
        if (frame.left >= width || frame.top >= height ||
            frame.width == 0 || frame.height == 0 ||
            frame.left + frame.width > width || 
            frame.top + frame.height > height) {
            LOGE("Invalid frame dimensions: %d,%d %dx%d in %dx%d canvas",
                 frame.left, frame.top, frame.width, frame.height, width, height);
            return;
        }
        
        uint8_t packed = mapped_data[offset++];
        frame.has_local_color = packed & 0x80;
        frame.interlaced = packed & 0x40;  // Capture interlaced flag
        
        if (frame.has_local_color) {
            uint8_t size_bits = (packed & 0x07) + 1;
            uint32_t color_table_size = 1 << size_bits;
            if (offset + 3 * color_table_size > data_size) return;
            frame.local_color_table = BuildColorTable(mapped_data + offset, color_table_size);
            offset += 3 * color_table_size;

    frame.color_table_size = color_table_size;

        }
        
        if (offset >= data_size) return;
        frame.min_code_size = mapped_data[offset++];
        
        // Read LZW data blocks
        size_t lzw_start = offset;
        while (offset < data_size && mapped_data[offset] != 0) {
            uint8_t block_size = mapped_data[offset++];
            if (offset + block_size > data_size) break;
            for (int i = 0; i < block_size; i++) {
                frame.lzw_data.push_back(mapped_data[offset++]);
            }
        }
        if (offset < data_size && mapped_data[offset] == 0) offset++;
        
        // Apply extension settings //you can set delay here
        int HandlerDelay=1;
        frame.delay = current_delay_; // > 0 ? current_delay_* HandlerDelay : 2;
        frame.disposal = current_disposal_?current_disposal_:1;
        //frame.disposal = current_disposal_;
        frame.transparent_index = transparent_index_;
        
        
                // Validate frame dimensions
        if (frame.left >= width || frame.top >= height ||
            frame.width == 0 || frame.height == 0 ||
            frame.left + frame.width > width || 
            frame.top + frame.height > height) {
            LOGE("Invalid frame dimensions: %d,%d %dx%d in %dx%d canvas",
                 frame.left, frame.top, frame.width, frame.height, width, height);
            return;
        }
        
        frames.push_back(frame);
        current_delay_=2;
        current_disposal_=1;
        transparent_index_=0xff;
    }

    void parse_frames() {
        size_t offset = 13 + 3 * global_color_table_size;
        
        while (offset < data_size) {
        	if(offset >= data_size) break;
            if (mapped_data[offset] == 0x2C) { // Image descriptor
                parse_image_frame(offset);
            } 
            else if (mapped_data[offset] == 0x21) { // Extension
                offset += parse_extension(offset);
            }     
            else if (mapped_data[offset] == 0x3B) { // Trailer
                break;
            } 
            else {
                offset++;
            }
        }
    }
    
    
size_t parse_extension(size_t offset) {
    if (offset >= data_size || mapped_data[offset] != 0x21) return 1;
    const uint8_t extension_type = mapped_data[offset + 1];
    size_t bytes_processed = 2;
    LOGI("Extension type %d", extension_type);
    
    switch (extension_type) {
        case 0xF9: { // Graphics Control Extension
            if (offset + 6 >= data_size) break;
            const uint8_t block_size = mapped_data[offset + 2];
            if (block_size != 4) break;
            
            const uint8_t packed = mapped_data[offset + 3];
            current_delay_ = mapped_data[offset + 4] | (mapped_data[offset + 5] << 8);
            current_disposal_ = (packed >> 2) & 0x07;
            if(current_disposal_  == 0)current_disposal_=1;
            transparent_index_ = (packed & 0x01) ? mapped_data[offset + 6] : 0xFF;
            
            bytes_processed += block_size + 1;
            break;
        }

        case 0xFF: { // Application Extension
            // Check we have at least the block size
            if (offset + 3 >= data_size) break;
            const uint8_t block_size = mapped_data[offset + 2];
            bytes_processed += 1; // For block_size byte
            
            // Check we have enough data for the identifier
            if (offset + bytes_processed + block_size > data_size) break;
            
            // Check for "NETSCAPE2.0" identifier
            if (block_size == 11 && 
                std::memcmp(mapped_data + offset + 3, "NETSCAPE2.0", 11) == 0) {
                
                bytes_processed += block_size;
                
                // Process data sub-blocks
                while (offset + bytes_processed < data_size) {
                    // Get next sub-block size
                    const uint8_t sub_size = mapped_data[offset + bytes_processed++];
                    
                    // Terminator block
                    if (sub_size == 0) break;
                    
                    // Check we have enough data for this sub-block
                    if (offset + bytes_processed + sub_size > data_size) break;
                    
                    // Check for loop count sub-block (size=3, ID=1)
                    if (sub_size == 3) {
                        if (mapped_data[offset + bytes_processed] == 0x01) {
                            loop_count_ = mapped_data[offset + bytes_processed + 1] | 
                                        (mapped_data[offset + bytes_processed + 2] << 8);
                        }
                    }
                    
                    bytes_processed += sub_size;
                }
                // Ensure we consume the terminator if present
                if (offset + bytes_processed < data_size && 
                    mapped_data[offset + bytes_processed] == 0) {
                    bytes_processed++;
                }
                break;
            } else {
                // Not NETSCAPE2.0 - skip the identifier
                bytes_processed += block_size;
                
                // Skip all data sub-blocks
                while (offset + bytes_processed < data_size) {
                    const uint8_t sub_size = mapped_data[offset + bytes_processed++];
                    if (sub_size == 0) break;
                    if (offset + bytes_processed + sub_size > data_size) break;
                    bytes_processed += sub_size;
                }
                // Skip terminator if present
                if (offset + bytes_processed < data_size && 
                    mapped_data[offset + bytes_processed] == 0) {
                    bytes_processed++;
                }
            }
            break;
        }

        default: { 
            // Skip unsupported extensions with bounds checking
            while (offset + bytes_processed < data_size && 
                   mapped_data[offset + bytes_processed] != 0) {
                uint8_t block_size = mapped_data[offset + bytes_processed++];
                if (offset + bytes_processed + block_size > data_size) {
                    bytes_processed = data_size - offset;
                    break;
                }
                bytes_processed += block_size;
            }
            if (offset + bytes_processed < data_size && 
                mapped_data[offset + bytes_processed] == 0) {
                bytes_processed++;
            }
            break;
        }
    }
    return bytes_processed;
}


    static void lzw_decode_free(LZWContext* ctx) {
        if (ctx) {
            std::free(ctx->prefix_table);
            std::free(ctx->suffix_stack);
        }
    }


static int lzw_decode_init(LZWContext* ctx, const uint8_t* data, size_t size, int min_code_size) {
    if (!ctx || !data || size < 1 || min_code_size < 2 || min_code_size > LZW_MAX_CODE_SIZE) 
        return -1;
    
    ctx->input_start = data;
    ctx->input_end = data + size;
    ctx->input_ptr = data;
    ctx->code_size = static_cast<uint8_t>(min_code_size);
    ctx->current_code_size = ctx->code_size + 1;
    ctx->clear_code = 1 << ctx->code_size;
    ctx->end_code = ctx->clear_code + 1;
    
    // Use calloc to ensure zero initialization
    ctx->prefix_table = static_cast<uint16_t*>(calloc(LZW_TABLE_SIZE, sizeof(uint16_t)));
    ctx->suffix_stack = static_cast<uint8_t*>(calloc(LZW_STACK_SIZE, sizeof(uint16_t)));
    //ctx->suffix_stack = static_cast<uint8_t*>(malloc(LZW_STACK_SIZE));
    
    
    if (!ctx->prefix_table || !ctx->suffix_stack) {
    	LOGE("LZW allocation failed");
        free(ctx->prefix_table);
        free(ctx->suffix_stack);
        return -1;
    } 

    ctx->stack_ptr = ctx->suffix_stack;
    ctx->next_code = ctx->end_code + 1;
    ctx->table_size = 1 << ctx->current_code_size;
    ctx->max_code = ctx->table_size - 1;
    ctx->current_bits = 0;
    ctx->bits_left = 0;
    ctx->prev_code = 0xFFFF;
    ctx->first_code = 0xFFFF;
    return 0;
}

static int lzw_decode(LZWContext* ctx, uint8_t* output, size_t output_size) {
    if (!ctx || !output) return -1;
    
    uint8_t* output_end = output + output_size;
    uint8_t* stack_base = ctx->suffix_stack;
    uint8_t* stack_end = stack_base + LZW_STACK_SIZE;
    uint16_t* prefix_table = ctx->prefix_table;
    
    while (output < output_end) {
           // Accumulate bits with 64-bit safety
           while (ctx->bits_left < ctx->current_code_size) {
               if (ctx->input_ptr >= ctx->input_end) {
                   return output_size - (output_end - output);
               }
               // Use 64-bit shift
               ctx->current_bits |= static_cast<uint64_t>(*ctx->input_ptr++) << ctx->bits_left;
               ctx->bits_left += 8;
           }
           
           
           // Extract code (mask stays 16-bit safe)
           const uint16_t code = static_cast<uint16_t>(ctx->current_bits & ((1 << ctx->current_code_size) -1));
           if(code >= LZW_TABLE_SIZE)
           {
           LOGE("Invalid LZW Code: %d (max: %d)", code, LZW_TABLE_SIZE);
           return -1;
           }
           
           ctx->current_bits >>= ctx->current_code_size;
           ctx->bits_left -= ctx->current_code_size ;

        
        if (code == ctx->end_code) break;
        if (code == ctx->clear_code) {
            ctx->current_code_size = ctx->code_size + 1;
            ctx->table_size = 1 << ctx->current_code_size;
            ctx->max_code = ctx->table_size - 1;
            ctx->next_code = ctx->end_code +1 ;
            ctx->prev_code = 0xFFFF;
            ctx->first_code = 0xFFFF;
            continue;
        }


        
        if(code >= LZW_TABLE_SIZE || code > ctx->next_code )
        {return -1;}

        uint16_t current_code = code;
        uint8_t* stack_ptr = ctx->suffix_stack;
       
        
        if (code == ctx->next_code) {
           // if (ctx->prev_code == 0xFFFF) return -1;
            if (stack_ptr >= stack_end ) return -1;
            *stack_ptr++ = static_cast<uint8_t>(ctx->first_code);
            current_code = ctx->prev_code;
        }
        
        
        
        while (current_code >= ctx->clear_code) {
            if (current_code >= LZW_TABLE_SIZE) return -1;
            if (stack_ptr >= stack_end ) return -1;
            *stack_ptr++ = ctx->suffix_stack[current_code];
            current_code = prefix_table[current_code];
        }
        
        
        if (stack_ptr >= stack_end) return -1;
        const uint8_t first_char = static_cast<uint8_t>(current_code);
        *stack_ptr++ = first_char;
        

        if (ctx->prev_code != 0xFFFF && ctx->next_code < LZW_TABLE_SIZE) {
            prefix_table[ctx->next_code] = ctx->prev_code;
            ctx->suffix_stack[ctx->next_code] = first_char;
            
            
            ctx->next_code++;
            //*
            if (ctx->next_code > ctx->max_code && ctx->current_code_size < LZW_MAX_CODE_SIZE) {
                ctx->current_code_size++;
                ctx->table_size = 1 << ctx->current_code_size;
                ctx->max_code = ctx->table_size - 1;
            }
           // */
        }
        
        ctx->first_code = current_code;
        ctx->prev_code = code;
        
        while (stack_ptr > stack_base && output < output_end) {
            *output++ = *--stack_ptr;
            
        }
    }

    return output_size - (output_end - output);
}

void restore_background(uint32_t* output, const Frame& frame, 
                       uint32_t canvas_width, uint32_t canvas_height) {
                  
    const uint32_t bg = bg_color;
    const uint32_t left = frame.left;
    const uint32_t top = frame.top;
    const uint32_t right = left + frame.width;
    const uint32_t bottom = top + frame.height;
    
    for (uint32_t y = top; y < bottom; y++) {
        if (y >= canvas_height) break;
        uint32_t* row = output + y * canvas_width;
        for (uint32_t x = left; x < right; x++) {
            if (x >= canvas_width) break;  // Add boundary check
            row[x] = bg;
        }
    }
}
    
   
void restore_previous_state(uint32_t* output, 
                           uint32_t width, uint32_t height) {
//*
    if (!prev_canvas) {
        // If there's no saved state, restore to background
        std::fill(output, output + width * height, bg_color);
        return;
    }
    const size_t total_pixels = width * height;
    memcpy(output, prev_canvas, total_pixels * sizeof(uint32_t));
//*/
}


void save_current_state(const uint32_t* output, 
                          uint32_t width, uint32_t height) {
        const size_t size = width * height * sizeof(uint32_t);
        if (!prev_canvas) {
            prev_canvas = static_cast<uint32_t*>(std::calloc(size,sizeof(uint32_t)));
        }
        if (prev_canvas) {
            std::memcpy(prev_canvas, output, size);
        }
    }

void handle_disposal(uint32_t frame_index, uint32_t* output, 
                        uint32_t width, uint32_t height) {
        if (frame_index == 0) {
            std::fill(output, output + width * height, bg_color);
            return;
        }

        const Frame& prev_frame = frames[frame_index - 1];
        switch (prev_frame.disposal) {
        
            case 1: // Keep current frame (do nothing)
                break;
            case 2: // Restore to background
                restore_background(output, prev_frame, width, height);
                break;
            case 3: // Restore to previous state
                if (saved_before_frame) {
                    std::memcpy(output, saved_before_frame.get(), 
                              width * height * sizeof(uint32_t));
                } else {
                    std::fill(output, output + width * height, bg_color);
                }
                break;
        }
    }


    
};

